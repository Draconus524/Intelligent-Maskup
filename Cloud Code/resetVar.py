import pyrebase

config = {
  "apiKey": "AUTH",
  "authDomain": "projectId.firebaseapp.com",
  "databaseURL": "URL",
  "storageBucket": "projectId.appspot.com",
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()

db.child("/").update({"Maskless":0})
