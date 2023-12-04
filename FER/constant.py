import os


CURRENT_PATH = 'D:/MasterProject/PYproject/FER/'

RELATIVE_DETECT_UPLOAD_PATH = os.path.join('data', 'detect', 'upload')

RELATIVE_DETECT_RESULT_PATH = os.path.join('data', 'detect', 'result')

DETECT_UPLOAD_PATH = os.path.join(CURRENT_PATH, RELATIVE_DETECT_UPLOAD_PATH)

DETECT_RESULT_PATH = os.path.join(CURRENT_PATH, RELATIVE_DETECT_RESULT_PATH)


RELATIVE_DISCERN_UPLOAD_PATH = os.path.join("data", "discern", "upload")

RELATIVE_DISCERN_RESULT_PATH = os.path.join("data", "discern", "result")

DISCERN_UPLOAD_PATH = os.path.join(CURRENT_PATH, RELATIVE_DISCERN_UPLOAD_PATH)

DISCERN_RESULT_PATH = os.path.join(CURRENT_PATH, RELATIVE_DISCERN_RESULT_PATH)


RELATIVE_ADMIN_LEARN_UPLOAD_PATH = os.path.join("/data", "/admin", "/upload")

ADMIN_LEARN_UPLOAD_PATH = os.path.join(CURRENT_PATH, RELATIVE_ADMIN_LEARN_UPLOAD_PATH)

TRAINER_PATH = os.path.join(CURRENT_PATH, "/model")



CASCADE_CLASSIFIER_FILE = os.path.join(CURRENT_PATH, "config", "haarcascade_frontalface_default.xml")


UUID_CHARS = ("a", "b", "c", "d", "e", "f","g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
              "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5","6", "7",
              "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I","J", "K", "L", "M", "N", "O",
              "P", "Q", "R", "S", "T", "U", "V","W", "X", "Y", "Z")


UUID_LEN = 8



ALLOWED_IMG_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'fersys',
    'user': 'root',
    'password': 'root'
}