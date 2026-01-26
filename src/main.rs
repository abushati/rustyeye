use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use ndarray::Array3;
use pyo3::ffi::c_str;
use std::ffi::CString;
fn main() -> PyResult<()> {
    Python::initialize();
    Python::with_gil(|py| {

        let code = std::fs::read_to_string("hailo_infer.py")?;
        // Load Python module

        let hailo_module = PyModule::from_code(
            py,
            CString::new(code)?.as_c_str(),
            c_str!("hailo_infer.py"),
            c_str!("hailo_infer"),
        )?;

        // Create dummy frame 160x120x3
        let frame = Array3::<f32>::zeros((160, 120, 3));
        let frame: Vec<f32> = vec![0.0; 2076672];
        // frame.to_degrees();
        // // Convert to numpy array
        let np = py.import("numpy")?;
        let array_fn = np.getattr("array")?;
        let py_frame = array_fn.call1((frame,))?;

        // Call Python function (single argument wrapped in 1-tuple)
        // hailo_module.call_method1("new", (py_frame,))?;
        let m = hailo_module.getattr("run_inference").unwrap();
        let output = m.call1((py_frame,));

        // let output = hailo_module.call_method1("run_inference",
        //                                         (py_frame,)).unwrap();
        println!("{:?}", output);

        // Extract output to Vec<f32>
        // let output_vec: Vec<f32> = output.extract()?;
        // println!("Output length: {}", output_vec.len());

        Ok(())
    })
}





// use axum::{Router, routing::get, body::Body, response::IntoResponse};
// use async_stream::stream;
// use bytes::Bytes;
// use image::{GrayImage, Luma};
// use std::{process::{Command, Child}, sync::{Arc, Mutex}, time::Duration, io::{Read, Write}, fs, os};
// use std::cmp::PartialEq;
// use std::path::Path;
// use chrono::{Local, Timelike, Utc};
// use std::process::Stdio;
// use tokio::sync::mpsc;
// use tokio::time::Instant;
// use serde::Deserialize;
// use lazy_static::lazy_static;
// use axum::extract::ws::{WebSocket, Message};
// use tokio::fs::File;
// use reqwest::Client;
// use tokio_util::codec::{BytesCodec, FramedRead};
// use reqwest::multipart;
// /* ================= CONFIG ================= */
//
// const PIXEL_NOISE_THRESHOLD: u8 = 8;   // ignore jpeg/ISP jitter
// const IDLE_FPS: u32 = 1;
//
// const ACTIVE_WIDTH: u32 = 640;
// const ACTIVE_HEIGHT: u32 = 320;
// const ACTIVE_FPS: u32 = 30;
//
// #[derive(Deserialize)]
// struct VideoConfig {
//     framerate: u32,
//     state_change_cooldown: u32,
//     motion_ratio_threshold: f32,
//     log_level: u32,
// }
//
// #[derive(Deserialize)]
// struct RecorderConfig {
//     framerate: u32,
//     base_recording_folder: String,
//     seperator_folder: String,
//     file_name_format: String,
//     file_path: String
// }
// #[derive(Deserialize)]
// pub struct Config {
//     video: VideoConfig,
//     recorder: RecorderConfig,
// }
// /* ================= SHARED STATE ================= */
//
// #[derive(Debug, PartialEq, Copy, Clone)]
// enum CameraState {
//     Idle,
//     MotionDetected,
// }
//
// struct Frames {
//     rgb: Option<Vec<u8>>,
//     diff: Option<Vec<u8>>,
//     jpeg: Option<Vec<u8>>
// }
//
// struct Camera {
//     width: u32,
//     height: u32,
//     fps: Arc<Mutex<u32>>,
//     state: CameraState,
//     process: Option<Child>,
//     frames: Arc<Mutex<Frames>>,
//     recorder_sender: mpsc::UnboundedSender<Vec<u8>>,
// }
//
//
// lazy_static! {
//     pub static ref CONFIG: Config = {
//         let contents = fs::read_to_string("config.toml").expect("Failed to read config.toml");
//         toml::from_str(&contents).expect("Failed to parse config.toml")
//     };
// }
// /* ================= CAMERA ================= */
//
// impl Camera {
//     fn new() -> (Self, Recorder) {
//         let (tx, rx) = mpsc::unbounded_channel::<Vec<u8>>();
//         let fps = Arc::new(Mutex::new(IDLE_FPS));
//         (
//             Self {
//                 width: ACTIVE_WIDTH,
//                 height: ACTIVE_HEIGHT,
//                 fps: fps.clone(),
//                 state: CameraState::Idle,
//                 process: None,
//                 frames: Arc::new(Mutex::new(Frames { rgb: None, diff: None, jpeg: None })),
//                 recorder_sender: tx,
//             },
//             Recorder::new(rx, fps.clone())
//         )
//     }
//
//     async fn run(&mut self) {
//         loop {
//             if self.process.is_none() {
//                 self.spawn_camera().await;
//                 continue;
//             }
//             let motion = self.capture_loop().await;
//             let utc_string = Utc::now().to_rfc3339();
//
//             match motion {
//                 CameraState::MotionDetected => {
//                     println!("⚡ Motion detected → switching to ACTIVE at {:?}", utc_string);
//                     self.restart(ACTIVE_FPS, motion).await;
//                 }
//                 CameraState::Idle => {
//                     println!("⚡ Motion not detected → switching to IDLE at {:?}", utc_string);
//                     self.restart(IDLE_FPS, motion).await;
//                 }
//             }
//         }
//     }
//
//     async fn spawn_camera(&mut self) {
//         let child = Command::new("rpicam-vid")
//             .args([
//                 "-t", "0",
//                 "--codec", "mjpeg",
//                 "--width", &self.width.to_string(),
//                 "--height", &self.height.to_string(),
//                 "--framerate", &self.fps.lock().unwrap().to_string(),
//                 "autofocus-mode", "manual",
//                 "lens-position", "0.0",
//                 "--inline",
//                 "--listen",
//                 "--info-text", "",
//                 "--nopreview",
//                 "-o", "-",
//             ])
//             .stdout(Stdio::piped())
//             .stderr(Stdio::null())
//             .spawn()
//             .expect("failed to start rpicam-vid");
//
//         self.process = Some(child);
//         println!("📷 Camera running {}x{} @ {}fps", self.width, self.height, self.fps.lock().unwrap().to_string());
//     }
//
//     async fn restart(&mut self, fps: u32, camera_state: CameraState){
//         if let Some(mut p) = self.process.as_mut() {
//             let _ = p.kill();
//             self.process = None;
//
//         }
//         *self.fps.lock().unwrap() = fps;
//         self.state = camera_state;
//
//         self.spawn_camera().await;
//
//     }
//
//     async fn capture_loop(&mut self) -> CameraState {
//         let mut child = self.process.as_mut().unwrap();
//         let mut stdout = child.stdout.take().unwrap();
//
//         let tx = self.recorder_sender.clone();
//         let frames = self.frames.clone();
//         let mut state_change_cooldown = None;
//
//         // local state copy (do not use self.state inside spawn_blocking)
//         let mut state = self.state;
//
//         let handle: tokio::task::JoinHandle<CameraState> = tokio::task::spawn_blocking(move || {
//             let mut buf = [0u8; 4096];
//             let mut acc = Vec::<u8>::new();
//             let mut prev: Option<GrayImage> = None;
//             let mut motion_history = vec![true; 1000];
//
//             loop {
//                 if CONFIG.video.log_level == 0 {
//                     println!("DEBUGGG");
//                 }
//                 let n = stdout.read(&mut buf).unwrap();
//                 if n == 0 { continue; }
//
//                 acc.extend_from_slice(&buf[..n]);
//
//                 while let Some(end) = acc.windows(2).position(|w| w == [0xFF, 0xD9]) {
//                     if state_change_cooldown.is_none() {
//                         state_change_cooldown = Some(Instant::now());
//                     }
//                     let frame = acc.drain(..end + 2).collect::<Vec<u8>>();
//
//                     // Send JPEG
//                     if state == CameraState::MotionDetected {
//                         let _ = tx.send(frame.clone());
//                     }
//                     // Store latest frame for HTTP
//                     {
//                         let mut f = frames.lock().unwrap();
//                         f.jpeg = Some(frame.clone());
//                     }
//
//                     // Motion diff
//                     let img = match image::load_from_memory(&frame) {
//                         Ok(i) => i.resize(160, 120, image::imageops::FilterType::Nearest).to_luma8(),
//                         Err(_) => continue,
//                     };
//
//                     let motion_ratio = if let Some(prev) = &prev {
//                         frame_diff(prev, &img)
//                     } else {
//                         0.0
//                     };
//
//                     prev = Some(img);
//                     // println!("{}", motion_ratio);
//                     let motion_detected = motion_ratio > CONFIG.video.motion_ratio_threshold;
//                     motion_history.push(motion_detected);
//
//                     if state_change_cooldown.as_mut().unwrap().elapsed() < Duration::from_secs(CONFIG.video.state_change_cooldown as u64) {
//                         // println!("{:?}", state_change_cooldown.as_mut().unwrap().elapsed());
//                         continue;
//                     }
//
//                     if motion_detected && state != CameraState::MotionDetected {
//                         state = CameraState::MotionDetected;
//                         tx.send(vec![255,255,255,255]).unwrap();
//                         return state; // return CameraState ✅
//                     }
//
//                     if state == CameraState::MotionDetected {
//                         if motion_history[motion_history.len().saturating_sub(310)..]
//                             .iter()
//                             .all(|&x| !x)
//                         {
//                             state = CameraState::Idle;
//                             tx.send(vec![255,255,255,1]).unwrap();
//                             return state; // return CameraState ✅
//                         }
//                     }
//                 }
//             }
//
//             // unreachable but type correct
//             state
//         });
//
//         handle.await.unwrap()
//     }
// }
//
// /* ================= RECORDER ================= */
//
// struct Recorder {
//     receiver: mpsc::UnboundedReceiver<Vec<u8>>,
//     mp4_writer: Option<Child>,
//     temp_mp4_writer: Option<Child>,
//     frame_rate: Arc<Mutex<u32>>,
// }
//
// #[derive(PartialEq)]
// enum Mode {
//     Temp,
//     Active
// }
//
// impl Recorder {
//     fn get_file_name() -> String {
//         let filename = Local::now().format(&CONFIG.recorder.file_name_format).to_string();
//         let seperator_folder = Local::now().format(&CONFIG.recorder.seperator_folder).to_string();
//         let template = &CONFIG.recorder.file_path;
//         let mut file = template
//             .replace("{base_recording_folder}", &CONFIG.recorder.base_recording_folder)
//             .replace("{seperator_folder}", &*seperator_folder)
//             .replace("{file_name_format}", &*filename);
//
//         let d = format!("{}/{}", &CONFIG.recorder.base_recording_folder, &seperator_folder);
//         if !Path::new(&d).exists() {
//             // create directory (and any missing parent folders)
//             fs::create_dir_all(&d).expect("Failed to create directory");
//             println!("Created directory: {}", &d);
//         } else {
//             println!("Directory already exists: {}", &d);
//         }
//
//         let mut count = 1;
//         loop {
//             if !Path::new(&file).exists() {
//                 break;
//             }
//             let mut v = file.split(".").collect::<Vec<&str>>();
//             let n = &format!("{}_{}", v[0], count.to_string());
//             v[0] = n;
//             file = v.join(".");
//             count += 1;
//         }
//         file
//     }
//
//     fn start_ffmpeg(mode: Mode) -> Child {
//         let mut file = "".to_string();
//         if mode == Mode::Active {
//             file = Recorder::get_file_name();
//         } else {
//             file = "temp_file.mp4".to_string();
//         }
//
//
//         // println!("Video file {file}");
//
//         let child = Command::new("ffmpeg")
//             .args([
//                 "-y",
//                 "-loglevel", "error",
//
//                 // input: MJPEG frames from stdin
//                 "-f", "mjpeg",
//                 // "-framerate", "30",
//                 "-i", "pipe:0",
//
//                 // 🔹 BURN TIMESTAMP ON VIDEO
//                 "-vf",
//                 "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
// text='%{localtime}':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.4",
//
//                 // video encoding (cold storage)
//                 "-c:v", "libx264",
//                 "-preset", "slow",
//                 "-crf", "30",
//                 "-pix_fmt", "yuv420p",
//
//                 // important for MP4
//                 "-movflags", "+faststart",
//                 "-strftime", "1",
//
//                 // output pattern
//                 file.as_str(),
//             ])
//             .stdin(Stdio::piped())
//             .stdout(Stdio::null())
//             .stderr(Stdio::inherit())
//             .spawn()
//             .expect("failed to start ffmpeg");
//         child
//     }
//
//     async fn start(&mut self){
//         let mut pervous_hour = None;
//         println!("Starting file working");
//         loop {
//             let now_time = chrono::Local::now();
//             if pervous_hour.is_none() {
//                 pervous_hour = Some(now_time.hour());
//                 if self.mp4_writer.is_none() {
//                     self.mp4_writer = Some(Recorder::start_ffmpeg(Mode::Active));
//                     self.recorder_task().await;
//                 }
//             } else {
//                 if pervous_hour.as_ref().unwrap().ne(&now_time.hour()) {
//                     self.mp4_writer.as_mut().unwrap().kill().unwrap();
//                     self.mp4_writer = Some(Recorder::start_ffmpeg(Mode::Active));
//                     self.recorder_task().await;
//                     pervous_hour = Some(now_time.hour());
//                 } else {
//                     tokio::time::sleep(Duration::from_secs(1)).await;
//                 }
//             }
//         }
//
//     }
//
//     fn new(receiver: mpsc::UnboundedReceiver<Vec<u8>>, frame_rate:Arc<Mutex<u32>>) -> Self {
//         Self { receiver, mp4_writer: None , temp_mp4_writer: None, frame_rate }
//     }
//
//     fn start_temp_recorder(& mut self) {
//         self.temp_mp4_writer = Some(Self::start_ffmpeg(Mode::Temp));
//     }
//
//     async fn send_to_discord(&self) {
//         println!("Sending to discord");
//         let channel_id = "1464459882207117334";
//         let bot_token = "MTQ2NDQ1ODQwNjQzMjYwODMyOQ.G1lzZR.F5VyPrO0pNrSxfKk87Ugch75fNANiTqTzkLCys"; // rotate it if leaked
//         let message_content = "🚨 Motion detected on Camera 1";
//         let video_path = "temp_file.mp4";
//
//         let url = format!("https://discord.com/api/v10/channels/{}/messages", channel_id);
//
//         // Prepare multipart (file + payload)
//         let file = File::open(video_path).await.unwrap();
//         let file_stream = FramedRead::new(file, BytesCodec::new());
//         let file_part = multipart::Part::stream(reqwest::Body::wrap_stream(file_stream))
//             .file_name("my_video.mp4")
//             .mime_str("video/mp4").unwrap();
//
//         let form = multipart::Form::new()
//             .text("content", message_content.to_string())
//             .part("file", file_part);
//
//         // Send POST request
//         let client = Client::new();
//         let res = client.post(&url)
//             .header("Authorization", format!("Bot {}", bot_token))
//             .multipart(form)
//             .send()
//             .await;
//
//         // Check result
//
//         let text = res.unwrap().text().await;
//         println!("Failed: | {}", text.unwrap());
//
//     }
//
//     async fn end_temp_recorder(& mut self) {
//         self.temp_mp4_writer.as_mut().unwrap().stdin.take();
//         self.temp_mp4_writer.as_mut().unwrap().wait();
//         self.temp_mp4_writer.as_mut().unwrap().kill().unwrap();
//         self.send_to_discord().await;
//     }
//
//
//     async fn recorder_task(&mut self) {
//         println!("🎬 Recorder started");
//
//         while let Some(message) = self.receiver.recv().await {
//             // Handle control messages first
//             if message.as_slice() == &[255, 255, 255, 255] {
//                 println!("Enabling temp mode");
//                 self.start_temp_recorder();
//                 continue; // skip writing this frame
//             } else if message.as_slice() == &[255, 255, 255, 1] {
//                 println!("Ending temp mode");
//                 self.end_temp_recorder().await;
//                 continue; // skip writing this frame
//             }
//
//             // Write to temp recorder first (if active)
//             if let Some(temp_writer) = self.temp_mp4_writer.as_mut() {
//                 if let Some(stdin) = temp_writer.stdin.as_mut() {
//                     let _ = stdin.write_all(&message);
//                 }
//             }
//
//             // Write to main recorder
//             {
//                 if let Some(main_writer) = self.mp4_writer.as_mut() {
//                     if let Some(stdin) = main_writer.stdin.as_mut() {
//                         let _ = stdin.write_all(&message);
//                         let _ = stdin.flush(); // optional for low-latency
//                     }
//                 }
//             }
//         }
//
//         // Finish main recorder
//         if let Some(main_writer) = self.mp4_writer.as_mut() {
//             let _ = main_writer.stdin.take(); // signal EOF
//             let _ = main_writer.wait();       // wait for FFmpeg to finish
//         }
//
//         println!("🎬 Recording finished");
//     }
// }
//
// /* ================= HTTP STREAM ================= */
//
//
// async fn stream_mjpeg(
//     frames: Arc<Mutex<Frames>>,
//     diff: bool,
// ) -> impl IntoResponse {
//
//     let body = stream! {
//         let mut count: u64 = 0;
//
//         loop {
//             let jpeg = {
//                 let f = frames.lock().unwrap();
//                 if diff { f.diff.clone() } else { f.jpeg.clone() }
//             };
//
//             if let Some(j) = jpeg {
//                 count += 1;
//
//                 let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
//
//                 yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(
//                     format!(
//                         "--frame\r\n\
//                          Content-Type: image/jpeg\r\n\
//                          Content-Length: {}\r\n\
//                          X-Timestamp: {}\r\n\
//                          X-Frame-Count: {}\r\n\r\n",
//                         j.len(),
//                         timestamp,
//                         count
//                     )
//                 ));
//
//                 yield Ok(Bytes::from(j));
//                 yield Ok(Bytes::from("\r\n"));
//             }
//
//             tokio::time::sleep(Duration::from_millis(50)).await;
//         }
//     };
//
//     (
//         [("Content-Type", "multipart/x-mixed-replace; boundary=frame")],
//         Body::from_stream(body),
//     )
// }
//
// /* ================= IMAGE HELPERS ================= */
//
// fn frame_diff(
//     prev: &GrayImage,
//     curr: &GrayImage,
// ) -> f32 {
//     let mut diff = GrayImage::new(curr.width(), curr.height());
//
//     let mut changed = 0u32;
//     let total = curr.width() * curr.height();
//
//     for (x, y, p) in curr.enumerate_pixels() {
//         let d = prev.get_pixel(x, y)[0].abs_diff(p[0]);
//
//         if d > PIXEL_NOISE_THRESHOLD {
//             diff.put_pixel(x, y, Luma([255]));
//             changed += 1;
//         } else {
//             diff.put_pixel(x, y, Luma([0]));
//         }
//     }
//
//     changed as f32 / total as f32
// }
//
// /* ================= MAIN ================= */
//
// #[tokio::main]
// async fn main() {
//     let (mut cam, mut recorder) = Camera::new();
//     let frames = cam.frames.clone();
//
//     tokio::spawn(async move { cam.run().await });
//     tokio::spawn(async move { recorder.start().await });
//
//     let app = Router::new()
//         .route("/frame", get({
//             let f = frames.clone();
//             move || stream_mjpeg(f, false)
//         }))
//         .route("/diff", get({
//             let f = frames.clone();
//             move || stream_mjpeg(f, true)
//         }));
//
//     println!("🌐 http://PI_IP:3000/frame");
//     println!("🌐 http://PI_IP:3000/diff");
//
//     axum::serve( tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap(), app, ) .await .unwrap();
// }
//
