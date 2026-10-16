
import os
from parkocc.app import create_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"ParkOcc-AI prototype -> http://0.0.0.0:{port}")
    create_app().run(host="0.0.0.0", port=port, debug=False)
