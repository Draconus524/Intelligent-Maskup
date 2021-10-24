import pyrebase

config = {
  "apiKey": "AUTH",
  "authDomain": "projectId.firebaseapp.com",
  "databaseURL": "URL",
  "storageBucket": "projectId.appspot.com",
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()

masklessValue = db.child("/Maskless").get()
masklessValue = int(masklessValue.val())
masklessValue = masklessValue +1

db.child("/").update({"Maskless":masklessValue})

print(masklessValue)
