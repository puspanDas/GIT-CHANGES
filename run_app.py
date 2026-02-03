import subprocess
import time
import os
import signal
import sys
import webbrowser

def run_commands():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    frontend_dir = os.path.join(base_dir, "frontend")

    print(f"🚀 Starting TaskFlow...")

    # Start Backend
    print(f"🐍 Starting Backend Server...")
    # Using shell=False ensures we get the actual process handle
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=backend_dir,
        shell=False 
    )

    # Start Frontend
    print(f"⚛️  Starting Frontend Server...")
    # Use 'npm.cmd' on Windows, 'npm' on others
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir,
        shell=False # Try false to get process handle
    )

    print(f"✅ Services are running!")
    print(f"   Backend: http://localhost:8000")
    print(f"   Frontend: http://localhost:5173")
    print(f"Press Ctrl+C to stop all services.")

    # Give services a moment to start
    time.sleep(2)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:5173")
    except:
        pass

    try:
        while True:
            # Check if processes are still alive
            if backend_process.poll() is not None:
                print("❌ Backend process terminated unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process terminated unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        # Terminate processes
        if backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process.poll() is None:
            # On Windows, terminate() might not kill the whole tree if shell=True was used, 
            # but with shell=False it should be better. 
            # For npm, it spawns node. terminate() kills npm. node might survive?
            # Let's try simple terminate first.
            frontend_process.terminate()
            
        # Wait for them to exit
        try:
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Forcing kill...")
            if backend_process.poll() is None:
                backend_process.kill()
            if frontend_process.poll() is None:
                frontend_process.kill()
                
        sys.exit(0)

if __name__ == "__main__":
    run_commands()
