use axum::{Router, routing::get, body::Body, response::IntoResponse};
use async_stream::stream;
use bytes::Bytes;
use image::{ImageBuffer, Luma, RgbImage};
use std::{
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use std::process::Child;
use tokio::{net::TcpStream, io::AsyncReadExt};

/* ================= CONFIG ================= */

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

const IDLE_FPS: Duration = Duration::from_secs(1);   // 1 FPS idle
const ACTIVE_FPS: Duration = Duration::from_millis(33); // 30 FPS on motion
const MOTION_THRESHOLD: u64 = 120000;

/* ================= SHARED STATE ================= */

struct Frames {
    rgb: Option<Vec<u8>>,
    diff: Option<Vec<u8>>,
}

struct Camera {
    width: u32,
    height: u32,
    framerate: u32,
    process: Option<Child>,
    stream: Option<TcpStream>,
    frames: Arc<Mutex<Frames>>

}
/* ================= MAIN ================= */

impl Camera {
    fn new(width: u32, height: u32, framerate: u32) -> Self {
        let frames = Arc::new(Mutex::new(Frames {
            rgb: None,
            diff: None,
        }));
        Self { width, height, framerate, process:None , stream: None, frames: frames }
    }

    async fn start(&mut self) {
        let child = Command::new("rpicam-vid")
            .args([
                "-t", "0",
                "--codec", "mjpeg",
                "--width", &self.width.to_string(),
                "--height", &self.height.to_string(),
                "--framerate", &self.framerate.to_string(),
                "--inline",
                "--listen",
                "-o", "tcp://0.0.0.0:8554",
            ])
            .spawn()
            .expect("failed to start rpicam-vid");


        let stream = {
            let mut attempt = 0;
            loop {
                match TcpStream::connect("127.0.0.1:8554").await {
                    Ok(s) => {
                        println!("✅ Connected to camera TCP");
                        break s;
                    }
                    Err(e) => {
                        attempt += 1;
                        if attempt >= 5 {
                            panic!("❌ Camera TCP not ready after 5 attempts: {}", e);
                        }
                        println!(
                            "⏳ Camera TCP not ready (attempt {}/5), retrying...",
                            attempt
                        );
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        };
        self.stream = Some(stream);
        self.process = Some(child);
        loop {
            let restart = self.capture_loop().await;
            if restart {
                self.restart().await;
            }
        }

        println!("📷 Camera started ({}x{} @ {}fps)",
                 self.width, self.height, self.framerate);
    }

    async fn restart(&mut self) {
        self.process.take().unwrap().kill();
        self.framerate = 30;
        self.height = 320;
        self.width = 640;
        self.start().await;

    }
    async fn capture_loop(&mut self) -> bool{

        let mut buf = vec![0u8; 128 * 1024];
        let mut prev: Option<Vec<u8>> = None;
        let mut last = Instant::now();
        let mut active = false;
        let mut stream = self.stream.take().unwrap();
        loop {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 { continue; }

            let delay = if active { ACTIVE_FPS } else { IDLE_FPS };
            if last.elapsed() < delay { continue; }
            last = Instant::now();

            let img = match image::load_from_memory(&buf[..n]) {
                Ok(i) => i.to_rgb8(),
                Err(_) => continue,
            };

            let (motion, diff_img) = if let Some(p) = &prev {
                frame_diff(p, &img)
            } else {
                (0, blank_luma(img.width(), img.height()))
            };

            active = motion > MOTION_THRESHOLD;
            if active {
                self.restart().await;
                return true;
            }

            prev = Some(img.clone().into_raw());

            let rgb_jpeg = encode_jpeg_rgb(&img);
            let diff_jpeg = encode_jpeg_luma(&diff_img);

            let mut f = self.frames.lock().unwrap();
            f.rgb = Some(rgb_jpeg);
            f.diff = Some(diff_jpeg);
        }
    }
}

#[tokio::main]
async fn main() {

    let mut cam = Camera::new(WIDTH, HEIGHT, 1);
    // Background capture loop (ALWAYS RUNNING)
    let cf = cam.frames.clone();
    let app = Router::new()
        .route("/frame", get({
            let f = cf.clone();
            move || stream_mjpeg(f, false)
        }))
        .route("/diff", get({
            let f = cf.clone();
            move || stream_mjpeg(f, true)
        }));

    cam.start().await;

    // HTTP server


    println!("🌐 http://PI_IP:3000/frame");
    println!("🌐 http://PI_IP:3000/diff");

    axum::serve(
        tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap(),
        app,
    )
        .await
        .unwrap();
}


/* ================= HTTP STREAM ================= */

async fn stream_mjpeg(
    frames: Arc<Mutex<Frames>>,
    diff: bool,
) -> impl IntoResponse {
    let body = stream! {
        loop {
            let frame = {
                let f = frames.lock().unwrap();
                if diff { f.diff.clone() } else { f.rgb.clone() }
            };

            if let Some(jpeg) = frame {
                yield Ok::<Bytes, std::convert::Infallible>(
                    Bytes::from("--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                );
                yield Ok(Bytes::from(jpeg));
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };

    (
        [("Content-Type", "multipart/x-mixed-replace; boundary=frame")],
        Body::from_stream(body),
    )
}

/* ================= IMAGE UTILS ================= */

fn frame_diff(prev: &[u8], curr: &RgbImage)
              -> (u64, ImageBuffer<Luma<u8>, Vec<u8>>)
{
    let raw = curr.as_raw();
    let mut out = Vec::with_capacity(raw.len() / 3);
    let mut sum = 0u64;

    for i in (0..raw.len()).step_by(3) {
        let d =
            (prev[i] as i16 - raw[i] as i16).abs() +
                (prev[i+1] as i16 - raw[i+1] as i16).abs() +
                (prev[i+2] as i16 - raw[i+2] as i16).abs();

        let g = (d / 3).min(255) as u8;
        sum += g as u64;
        out.push(g);
    }

    (
        sum,
        ImageBuffer::from_raw(curr.width(), curr.height(), out).unwrap()
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

/* ================= AI HOOK ================= */

fn ai_hook(_img: &RgbImage, motion: u64) {
    println!("🧠 AI HOOK | motion={}", motion);

    // Drop-in points:
    // - Resize
    // - Normalize
    // - Send to YOLO / TFLite / AI HAT
}
