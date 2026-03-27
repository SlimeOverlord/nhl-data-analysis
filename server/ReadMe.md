# Flask Server

## Run the server

Install the dependecies

```bash
pip install -r requirements.txt
```

To run the Flask server, use the following command:

```bash
cd server
gunicorn --bind 0.0.0.0:<PORT> app:app
```

or

```bash
cd server
waitress-serve --listen=0.0.0.0:<PORT> app:app
```
