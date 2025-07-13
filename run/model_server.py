import subprocess
import os
import signal
import sys
import time


class ModelServerManager:
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def start_server(self):
        try:
            print("Starting the model server...")
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            print(f"Server started with PID {self.process.pid}")
        except Exception as e:
            print(f"Error starting the server: {e}")
            sys.exit(1)

    def stop_server(self):
        if self.process:
            print("Stopping the model server...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait()
                print("Server stopped successfully.")
            except Exception as e:
                print(f"Error stopping the server: {e}")
        else:
            print("No server running.")

    def monitor_server(self):
        try:
            while True:
                retcode = self.process.poll()
                if retcode is not None:
                    print("Server process has terminated.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring interrupted.")
            self.stop_server()


if __name__ == "__main__":
    cmd = "CUDA_VISIBLE_DEVICES=0 python utils/model_server.py --base_model_string=THUDM/chatglm3-6b"

    server_manager = ModelServerManager(cmd)
    try:
        server_manager.start_server()
        server_manager.monitor_server()
    except Exception as e:
        print(f"Unexpected error: {e}")
        server_manager.stop_server()