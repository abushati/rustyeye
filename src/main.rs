use axum::{Router, routing::get, body::Body, response::IntoResponse};
use async_stream::stream;
use bytes::Bytes;
use image::{ImageBuffer, Luma, RgbImage};
use std::{
    process::{Command, Child},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use std::cmp::PartialEq;

use chrono::Utc;
use std::process::Stdio;
use tokio::{net::TcpStream, io::AsyncReadExt};

/* ================= CONFIG ================= */

const PIXEL_NOISE_THRESHOLD: u8 = 8;   // ignore jpeg/ISP jitter
const MOTION_PIXEL_VALUE: u8 = 255;
const IDLE_WIDTH: u32 = 320;
const IDLE_HEIGHT: u32 = 240;
const IDLE_FPS: u32 = 1;

const ACTIVE_WIDTH: u32 = 640;
const ACTIVE_HEIGHT: u32 = 320;
const ACTIVE_FPS: u32 = 30;

const MOTION_THRESHOLD: u64 = 120_000;

/* ================= SHARED STATE ================= */

#[derive(Debug, PartialEq)]

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
    fps: u32,
    state: CameraState,
    process: Option<Child>,
    stream: Option<TcpStream>,
    frames: Arc<Mutex<Frames>>,
}

/* ================= CAMERA ================= */

impl Camera {
    fn new() -> Self {
        Self {
            width: IDLE_WIDTH,
            height: IDLE_HEIGHT,
            fps: IDLE_FPS,
            state: CameraState::Idle,
            process: None,
            stream: None,
            frames: Arc::new(Mutex::new(Frames { rgb: None, diff: None , jpeg: None })),
        }
    }

    async fn run(&mut self) {
        self.spawn_camera().await;

        loop {
            let motion = self.capture_loop().await;
            let utc_string = Utc::now().to_rfc3339();
            match motion {
                CameraState::MotionDetected => {
                    println!("⚡ Motion detected → switching to ACTIVE at {:?}", utc_string);
                    self.restart(ACTIVE_WIDTH, ACTIVE_HEIGHT, ACTIVE_FPS, motion).await;
                }
                CameraState::Idle => {
                    println!("⚡ Motion not detected → switching to IDLE at {:?}", utc_string);
                    self.restart(IDLE_WIDTH, IDLE_HEIGHT, IDLE_FPS, motion).await;
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
                "--framerate", &self.fps.to_string(),
                "--inline",
                "--listen",
                "--info-text", "",
                "--nopreview",
                "-o", "tcp://0.0.0.0:8554",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start rpicam-vid");

        let stream = connect_camera_tcp().await;

        self.process = Some(child);
        self.stream = Some(stream);

        println!("📷 Camera running {}x{} @ {}fps", self.width, self.height, self.fps);
    }

    async fn restart(&mut self, w: u32, h: u32, fps: u32, camera_state: CameraState) {
        if let Some(mut p) = self.process.take() {
            let _ = p.kill();
        }

        self.width = w;
        self.height = h;
        self.fps = fps;
        self.state = camera_state;

        self.spawn_camera().await;
    }

    async fn capture_loop(&mut self) -> CameraState {
        let mut buf = vec![0u8; 128 * 1024];
        let mut prev: Option<RgbImage> = None;
        let mut motion_history = vec![true; 1000];
        let start_time = Instant::now();


        let mut stream = self.stream.take().unwrap();

        loop {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 { continue; }

            let jpeg_frame = buf[..n].to_vec();

            // Store MJPEG directly
            {
                let mut f = self.frames.lock().unwrap();
                f.jpeg = Some(jpeg_frame.clone());
            }

            // Decode SMALL image only
            let img = match image::load_from_memory(&jpeg_frame) {
                Ok(i) => i.resize(160, 120, image::imageops::FilterType::Nearest).to_rgb8(),
                Err(_) => continue,
            };


            let (motion_ratio, diff) = if let Some(prev) = &prev {
                frame_diff(prev, &img)
            } else {
                (0.0, blank_luma(img.width(), img.height()))
            };
            prev = Some(img);
            let motion_detected = motion_ratio > 0.03;

            motion_history.push(motion_detected);

            {
                let mut f = self.frames.lock().unwrap();
                f.diff = Some(encode_jpeg_luma(&diff));
            }


            if motion_detected && self.state
                != CameraState::MotionDetected &&
                Instant::now() - start_time > Duration::from_secs(10) {
                    self.stream = Some(stream);
                    println!("Motion Detected value {}", motion_detected);
                    return CameraState::MotionDetected;
            }

            if self.state == CameraState::MotionDetected
                && motion_history[motion_history.len().
                saturating_sub(200)..]
                .iter()
                .all(|&x| x == false)
            {
                println!("{}", motion_history.len());
                self.stream = Some(stream);
                return CameraState::Idle;
            }
        }
    }
}

/* ================= MAIN ================= */

#[tokio::main]
async fn main() {
    let mut cam = Camera::new();
    let frames = cam.frames.clone();

    tokio::spawn(async move {
        cam.run().await;
    });

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

    axum::serve(
        tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap(),
        app,
    )
        .await
        .unwrap();
}

/* ================= TCP ================= */

async fn connect_camera_tcp() -> TcpStream {
    for i in 1..=5 {
        match TcpStream::connect("127.0.0.1:8554").await {
            Ok(s) => {
                println!("✅ Camera TCP connected");
                return s;
            }
            Err(_) => {
                println!("⏳ Waiting for camera TCP ({}/5)", i);
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
    panic!("❌ Camera TCP not available");
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
                yield Ok::<Bytes, std::convert::Infallible>(
                    Bytes::from("--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                );
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

/* ================= IMAGE ================= */

fn frame_diff(
    prev_rgb: &[u8],
    curr: &RgbImage,
) -> (f32, ImageBuffer<Luma<u8>, Vec<u8>>) {

    let raw = curr.as_raw();
    let mut diff = Vec::with_capacity(raw.len() / 3);

    let mut changed_pixels: u32 = 0;
    let total_pixels = (raw.len() / 3) as u32;

    for i in (0..raw.len()).step_by(3) {
        // Convert both frames to grayscale on the fly
        let prev_gray =
            (prev_rgb[i] as u16 * 30 +
                prev_rgb[i + 1] as u16 * 59 +
                prev_rgb[i + 2] as u16 * 11) / 100;

        let curr_gray =
            (raw[i] as u16 * 30 +
                raw[i + 1] as u16 * 59 +
                raw[i + 2] as u16 * 11) / 100;

        let d = (prev_gray as i16 - curr_gray as i16).abs() as u8;

        if d > PIXEL_NOISE_THRESHOLD {
            diff.push(MOTION_PIXEL_VALUE);
            changed_pixels += 1;
        } else {
            diff.push(0);
        }
    }

    let motion_ratio = changed_pixels as f32 / total_pixels as f32;

    (
        motion_ratio,
        ImageBuffer::from_raw(curr.width(), curr.height(), diff).unwrap()
    )
}

fn blank_luma(w: u32, h: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    ImageBuffer::from_pixel(w, h, Luma([0]))
}

fn encode_jpeg_rgb(img: &RgbImage) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut out)
        .encode_image(img)
        .unwrap();
    out
}

fn encode_jpeg_luma(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut out)
        .encode_image(img)
        .unwrap();
    out
}
