from flask import Flask
import yaml

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def helloWorld():

    return 'HELLO WORLD!'

@app.route("/getSeed", methods = ['GET'])
def getSeed():
    # read seed data from yml file
    with open("seeds.yml") as file:
        data = yaml.safe_load(file)

    # get top seed value
    seed = data[0]["seeds"][0]
    data[0]["seeds"].remove(seed)

    if(len(data[0]) <= 5):
        pass
        # GET MORE SEEDS

    # update yml file
    with open("seeds.yml", 'w') as file:
        yaml.dump(data, file)
    
    return seed

if __name__ == '__main__':
    app.run()



