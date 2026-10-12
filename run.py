from parkocc.app import create_app
if __name__ == "__main__":
    print("ParkOcc-AI prototype -> http://127.0.0.1:5000")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
