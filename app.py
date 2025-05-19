from flask import Flask, send_from_directory
from flask_cors import CORS # type: ignore
from routes import api_routes
import os

app = Flask(__name__, static_folder='../../frontend/build', static_url_path='/')
CORS(app)

# Register API routes
app.register_blueprint(api_routes)

# Serve frontend build
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Serve other static files (JS, CSS, images)
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
