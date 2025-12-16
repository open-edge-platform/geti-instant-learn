// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    env,
    process::{Child, Command},
    sync::{Arc, Mutex},
    time::Duration,
};
use tauri::RunEvent;
use reqwest::blocking::Client;

/// “geti-prompt-backend.exe” on Windows, “geti-prompt-backend” elsewhere.
fn backend_filename() -> &'static str {
    if cfg!(windows) {
        "geti-prompt-backend.exe"
    } else {
        "geti-prompt-backend"
    }
}

/// Spawns the side-car in the same folder as this executable.
fn spawn_backend() -> std::io::Result<Child> {
    // Locate the Tauri executable, then its parent folder
    let exe_path = env::current_exe().expect("failed to get current exe path");
    let exe_dir = exe_path
        .parent()
        .expect("failed to get parent directory of exe");

    // Build the full path to geti-prompt-backend.exe
    // Tauri build will have renamed the suffixed file to plain name next to the exe.
    let backend_path = exe_dir.join(backend_filename());

    log::info!("▶ Looking for backend side-car at {:?}", backend_path);
    let mut command = Command::new(&backend_path);
    command.env("CORS_ORIGINS", "http://tauri.localhost");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let child = command.spawn()?;

    // Wait for backend to be ready
    let client = Client::new();
    let max_attempts = 15;
    let delay = Duration::from_millis(1000);
    let url = "http://127.0.0.1:9100/health";

    for _ in 0..max_attempts {
        if client.get(url).send().is_ok() {
            log::info!("✅ Backend REST API is up at {}", url);
            break;
        }
        std::thread::sleep(delay);
    }

    log::info!("▶ Spawned backend: {:?}", backend_path);
    Ok(child)
}

fn main() {
    // Shared handle so we can kill it on exit
    let child_handle = Arc::new(Mutex::new(None));

    // Build the app
    let app = tauri::Builder::default()
        .setup({
            let child_handle = child_handle.clone();
            move |_app_handle| {
                let child = spawn_backend().expect("Failed to spawn python backend");
                *child_handle.lock().unwrap() = Some(child);
                Ok(())
            }
        })
        .invoke_handler(tauri::generate_handler![])
        .build(tauri::generate_context!())
        .expect("error building Tauri");

    // Run and on Exit make sure to kill the backend
    let exit_handle = child_handle.clone();
    app.run(move |_app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = exit_handle.lock().unwrap().take() {
                let _ = child.kill();
                log::info!("⛔ Backend terminated");
            }
        }
    });
}