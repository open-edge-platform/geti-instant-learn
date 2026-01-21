// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    env,
    net::{TcpListener},
    process::{Child, Command},
    sync::{Arc, Mutex},
};
use tauri::RunEvent;

/// “geti-prompt-backend.exe” on Windows, “geti-prompt-backend” elsewhere.
fn backend_filename() -> &'static str {
    if cfg!(windows) {
        "geti-prompt-backend.exe"
    } else {
        "geti-prompt-backend"
    }
}

/// Picks a free port by binding to port 0 and returns it.
fn pick_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind to address")
        .local_addr()
        .expect("failed to get local address")
        .port()
}

/// Spawns the side-car in the same folder as this executable.
fn spawn_backend(port: u16) -> std::io::Result<Child> {
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
    command.env("CORS_ORIGINS", "tauri://localhost,http://tauri.localhost");
    command.env("PORT", port.to_string());
    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    if cfg!(debug_assertions) {
        command.env("LOG_LEVEL", "DEBUG");
    } else {
        command.env("LOG_LEVEL", "INFO");
    }

    let child = command.spawn()?;

    log::info!("▶ Spawned backend: {:?} on port {}", backend_path, port);
    Ok(child)
}

fn main() {
    // Shared handle so we can kill it on exit
    let child_handle = Arc::new(Mutex::new(None));
    let port = pick_free_port();
    // Build the app
    let app = tauri::Builder::default()
        .setup({
            let child_handle = child_handle.clone();
            let port = port;
            move |_app_handle| {
                let child = spawn_backend(port).expect("Failed to spawn python backend");
                *child_handle.lock().unwrap() = Some(child);
                Ok(())
            }
        })
        .invoke_handler(tauri::generate_handler![get_public_api_url])
        .manage(AppState { port })
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

struct AppState {
    port: u16,
}

#[tauri::command]
fn get_public_api_url(state: tauri::State<AppState>) -> String {
    format!("http://localhost:{}", state.port)
}
