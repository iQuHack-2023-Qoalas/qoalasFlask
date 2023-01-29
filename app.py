from flask import Flask
import yaml
import threading

import qrandomtemp as qr
app = Flask(__name__)

# @app.route("/", methods = ['GET'])
# def helloWorld():

#     return 'HELLO WORLD!'

@app.route("/", methods = ['GET'])
def getSeed():
    try:
        # read seed data from yml file
        with open("seeds.yml") as file:
            data = yaml.safe_load(file)
            file.close()

        # get top seed value
        seed = data[0]["seeds"][0]
        data[0]["seeds"].remove(seed)

        if(len(data[0]['seeds']) <= 5):
            # GET MORE SEEDS
            x = threading.Thread(target = threadFunction)
            x.start()

        # update yml file
        with open("seeds.yml", 'w') as file:
            print(data)
            yaml.dump(data, file)
            file.close()
        
        return str(seed)
    
    except Exception as e: return str(e)

def threadFunction():
    qr.generateNumbers()

if __name__ == '__main__':
    app.run()