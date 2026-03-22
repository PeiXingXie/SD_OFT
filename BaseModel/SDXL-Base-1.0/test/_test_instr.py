"""
Quick test instructions (run from repo root after starting the daemon server):

1) Start server (in another terminal):
  python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_server.py --host 127.0.0.1 --port 6320 --device cuda --torch_dtype float16 --variant fp16

2) Run a few client calls:
  python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_client.py --host 127.0.0.1 --port 6320 --prompt "A capybara holding a sign that reads Hello World, in Pointillism Style." --out BaseModel/SDXL-Base-1.0/test/capybara-Pointillism.png
  python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_client.py --host 127.0.0.1 --port 6320 --prompt "a painting of a church with a clock tower, in Baroque Style." --out BaseModel/SDXL-Base-1.0/test/church-Baroque.png
"""