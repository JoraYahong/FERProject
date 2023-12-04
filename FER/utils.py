import constant
from uuid import uuid1
from flask import jsonify
import pymysql
# from dbutils.pooled_db import PooledDB, SharedDBConnection



def short_uuid():
    uuid = str(uuid1()).replace('-', '')
    short_id = ''
    for i in range(0, constant.UUID_LEN):
        sub = uuid[i * 4: i * 4 + 4]
        x = int(sub, 16)
        short_id += constant.UUID_CHARS[x % 0x3E]
    return short_id



def get_img_name(is_new=True, img_path=None, file_name=None):
    if is_new:
        filename = short_uuid() + "." + file_name.split('.')[1]
    else:
        if img_path is not None:
            print(img_path.split('.'))
            name_len = constant.UUID_LEN + len("." + img_path.split('.')[1])
            path_len = len(img_path)
            filename = img_path[path_len - name_len: path_len]
        elif file_name is not None:
            filename = file_name.split('.')[0]
    return filename



class result_util:
    def __init__(self):
        self.result = {"code": 0, "msg": '', "data": ''}

    def ok(self, msg="操作成功！", data=None):
        self.set_attr(0, msg, data)
        return jsonify(self.result)

    def error(self, msg="操作失败！", data=None):
        self.set_attr(-1, msg, data)
        return jsonify(self.result)

    def set_attr(self, code, msg, data):
        self.result["code"] = code
        self.result["msg"] = msg
        self.result["data"] = data



class mysql_PooledDB(object):

    def __init__(self):

        self.pool_db = PooledDB(

            creator=pymysql,

            maxconnections=3,

            mincached=2,

            maxcached=5,

            maxshared=3,

            blocking=True,

            setsession=[],

            ping=0

        )
        coon = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               password='123456',
                               database='fersys'
                               )
        cur = coon.cursor()


    def insert(self, sql, data):

        count = self.cur.execute(sql, data)

        self.coon.commit()
        return count


    def delete(self, sql):

        count = self.cur.execute(sql)

        self.coon.commit()
        return count

    def update(self, sql):

        count = self.cur.execute(sql)

        self.coon.commit()
        return count


    def select(self, sql, param=None):

        if param is None:
            self.cur.execute(sql)
        else:
            self.cur.execute(sql, param)
        return self.cur.fetchmany()


    def dispose(self):
        self.coon.close()
        self.cur.close()