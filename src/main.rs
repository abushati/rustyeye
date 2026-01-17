use axum::{Router, routing::get, body::Body, response::IntoResponse};
use async_stream::stream;
use bytes::Bytes;
use image::{GrayImage, ImageBuffer, Luma, RgbImage};
use std::{
    process::{Command, Child},
    sync::{Arc, Mutex},
    time::Duration,
    io::{Read, Write},
};
use chrono::Utc;
use std::process::Stdio;
use std::thread::sleep;
use tokio::sync::mpsc;
use tokio::time::Instant;
/* ================= CONFIG ================= */

const PIXEL_NOISE_THRESHOLD: u8 = 8;   // ignore jpeg/ISP jitter
const MOTION_PIXEL_VALUE: u8 = 255;
const IDLE_WIDTH: u32 = 320;
const IDLE_HEIGHT: u32 = 240;
const IDLE_FPS: u32 = 1;

const ACTIVE_WIDTH: u32 = 640;
const ACTIVE_HEIGHT: u32 = 320;
const ACTIVE_FPS: u32 = 30;

/* ================= SHARED STATE ================= */

#[derive(Debug, PartialEq, Copy, Clone)]
enum CameraState {
    Idle,
    MotionDetected,
}

struct Frames {
    rgb: Option<Vec<u8>>,
    diff: Option<Vec<u8>>,
    jpeg: Option<Vec<u8>>
}

struct Camera {
    width: u32,
    height: u32,
    fps: Arc<Mutex<u32>>,
    state: CameraState,
    process: Option<Child>,
    frames: Arc<Mutex<Frames>>,
    recorder_sender: mpsc::UnboundedSender<Vec<u8>>,
}

/* ================= CAMERA ================= */

impl Camera {
    fn new() -> (Self, Recorder) {
        let (tx, rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let fps = Arc::new(Mutex::new(IDLE_FPS));
        (
            Self {
                width: ACTIVE_WIDTH,
                height: ACTIVE_HEIGHT,
                fps: fps.clone(),
                state: CameraState::Idle,
                process: None,
                frames: Arc::new(Mutex::new(Frames { rgb: None, diff: None, jpeg: None })),
                recorder_sender: tx,
            },
            Recorder::new(rx, fps.clone())
        )
    }

    async fn run(&mut self) {
        loop {
            if self.process.is_none() {
                self.spawn_camera().await;
                continue;
            }
            let motion = self.capture_loop().await;
            let utc_string = Utc::now().to_rfc3339();

            match motion {
                CameraState::MotionDetected => {
                    println!("⚡ Motion detected → switching to ACTIVE at {:?}", utc_string);
                    self.restart(ACTIVE_FPS, motion).await;
                }
                CameraState::Idle => {
                    println!("⚡ Motion not detected → switching to IDLE at {:?}", utc_string);
                    self.restart(IDLE_FPS, motion).await;
                }
            }
        }
    }

    async fn spawn_camera(&mut self) {
        let child = Command::new("rpicam-vid")
            .args([
                "-t", "0",
                "--codec", "mjpeg",
                "--width", &self.width.to_string(),
                "--height", &self.height.to_string(),
                "--framerate", &self.fps.lock().unwrap().to_string(),
                "--inline",
                "--listen",
                "--info-text", "",
                "--nopreview",
                "-o", "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start rpicam-vid");

        self.process = Some(child);
        println!("📷 Camera running {}x{} @ {}fps", self.width, self.height, self.fps.lock().unwrap().to_string());
    }

    async fn restart(&mut self, fps: u32, camera_state: CameraState){
        if let Some(mut p) = self.process.as_mut() {
            let _ = p.kill();
            self.process = None;

        }
        *self.fps.lock().unwrap() = fps;
        self.state = camera_state;

        self.spawn_camera().await;

    }

    async fn capture_loop(&mut self) -> CameraState {
        let mut child = self.process.as_mut().unwrap();
        let mut stdout = child.stdout.take().unwrap();

        let tx = self.recorder_sender.clone();
        let frames = self.frames.clone();
        let state_change_cooldown = Instant::now();

        // local state copy (do not use self.state inside spawn_blocking)
        let mut state = self.state;

        let handle: tokio::task::JoinHandle<CameraState> = tokio::task::spawn_blocking(move || {
            let mut buf = [0u8; 4096];
            let mut acc = Vec::<u8>::new();
            let mut prev: Option<GrayImage> = None;
            let mut motion_history = vec![true; 1000];

            loop {
                let n = stdout.read(&mut buf).unwrap();
                if n == 0 { continue; }

                acc.extend_from_slice(&buf[..n]);

                while let Some(end) = acc.windows(2).position(|w| w == [0xFF, 0xD9]) {
                    let frame = acc.drain(..end + 2).collect::<Vec<u8>>();

                    // Send JPEG
                    let _ = tx.send(frame.clone());

                    // Store latest frame for HTTP
                    {
                        let mut f = frames.lock().unwrap();
                        f.jpeg = Some(frame.clone());
                    }

                    // Motion diff
                    let img = match image::load_from_memory(&frame) {
                        Ok(i) => i.resize(160, 120, image::imageops::FilterType::Nearest).to_luma8(),
                        Err(_) => continue,
                    };

                    let motion_ratio = if let Some(prev) = &prev {
                        frame_diff(prev, &img)
                    } else {
                        0.0
                    };

                    prev = Some(img);

                    let motion_detected = motion_ratio > 0.03;
                    motion_history.push(motion_detected);

                    if state_change_cooldown.elapsed() < Duration::from_secs(3) {
                        continue;
                    }

                    if motion_detected && state != CameraState::MotionDetected {
                        state = CameraState::MotionDetected;
                        return state; // return CameraState ✅
                    }

                    if state == CameraState::MotionDetected {
                        if motion_history[motion_history.len().saturating_sub(310)..]
                            .iter()
                            .all(|&x| !x)
                        {
                            state = CameraState::Idle;
                            return state; // return CameraState ✅
                        }
                    }
                }
            }

            // unreachable but type correct
            state
        });

        handle.await.unwrap()
    }
}

/* ================= RECORDER ================= */

struct Recorder {
    receiver: mpsc::UnboundedReceiver<Vec<u8>>,
    mp4_writer: Option<Child>,
    frame_rate: Arc<Mutex<u32>>,
}

impl Recorder {
    fn new(receiver: mpsc::UnboundedReceiver<Vec<u8>>, frame_rate:Arc<Mutex<u32>>) -> Self {
        Self { receiver, mp4_writer: None , frame_rate}
    }

    fn start_ffmpeg(&mut self) -> Child {
        let filename = format!("recording_{}.avi", Utc::now().format("%Y%m%d_%H%M%S"));
        println!("Video file {filename}");

        let child = Command::new("ffmpeg")
            .args([
                "-y",
                "-loglevel", "error",

                // input: MJPEG frames from stdin
                "-f", "mjpeg",
                "-framerate", &self.frame_rate.lock().unwrap().to_string(),
                "-i", "pipe:0",

                // video encoding (cold storage)
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "30",
                "-pix_fmt", "yuv420p",

                // segmentation
                "-f", "segment",
                "-segment_time", "5",
                "-reset_timestamps", "1",

                // important for MP4 segments
                "-movflags", "+faststart",
                "-strftime", "1",

                // output pattern
                "recording2_%Y%m%d_%H%M%S_%03d.mp4",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start ffmpeg");

        child
    }

    async fn recorder_task(&mut self) {
        println!("🎬 Recorder started");
        self.mp4_writer = Some(self.start_ffmpeg());
        let w = self.mp4_writer.as_mut().unwrap();

        while let Some(frame) = self.receiver.recv().await {
            if let Some(stdin) = w.stdin.as_mut() {
                let _ = stdin.write_all(&frame);
                stdin.flush().unwrap();
                // tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        let _ = w.stdin.take();
        let _ = w.wait();
        println!("🎬 Recording finished");
    }
}

/* ================= HTTP STREAM ================= */

async fn stream_mjpeg(
    frames: Arc<Mutex<Frames>>,
    diff: bool,
) -> impl IntoResponse {
    let body = stream! {
        loop {
            let jpeg = {
                let f = frames.lock().unwrap();
                if diff { f.diff.clone() } else { f.jpeg.clone() }
            };

            if let Some(j) = jpeg {
                yield Ok::<Bytes, std::convert::Infallible>(Bytes::from("--frame\r\nContent-Type: image/jpeg\r\n\r\n"));
                yield Ok(Bytes::from(j));
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };

    (
        [("Content-Type", "multipart/x-mixed-replace; boundary=frame")],
        Body::from_stream(body),
    )
}

/* ================= IMAGE HELPERS ================= */

fn frame_diff(
    prev: &GrayImage,
    curr: &GrayImage,
) -> f32 {
    let mut diff = GrayImage::new(curr.width(), curr.height());

    let mut changed = 0u32;
    let total = curr.width() * curr.height();

    for (x, y, p) in curr.enumerate_pixels() {
        let d = prev.get_pixel(x, y)[0].abs_diff(p[0]);

        if d > PIXEL_NOISE_THRESHOLD {
            diff.put_pixel(x, y, Luma([255]));
            changed += 1;
        } else {
            diff.put_pixel(x, y, Luma([0]));
        }
    }

    changed as f32 / total as f32
}


fn blank_luma(w: u32, h: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    ImageBuffer::from_pixel(w, h, Luma([0]))
}

fn encode_jpeg_rgb(img: &RgbImage) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut out).encode_image(img).unwrap();
    out
}

fn encode_jpeg_luma(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut out).encode_image(img).unwrap();
    out
}

/* ================= MAIN ================= */

#[tokio::main]
async fn main() {
    let (mut cam, mut recorder) = Camera::new();
    let frames = cam.frames.clone();

    tokio::spawn(async move { cam.run().await });
    tokio::spawn(async move { recorder.recorder_task().await });

    let app = Router::new()
        .route("/frame", get({
            let f = frames.clone();
            move || stream_mjpeg(f, false)
        }))
        .route("/diff", get({
            let f = frames.clone();
            move || stream_mjpeg(f, true)
        }));

    println!("🌐 http://PI_IP:3000/frame");
    println!("🌐 http://PI_IP:3000/diff");

    axum::serve( tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap(), app, ) .await .unwrap();
}

