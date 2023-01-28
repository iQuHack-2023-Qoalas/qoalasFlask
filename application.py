import Flask

application = Flask()

@application.route("/getSeed", methods = ['GET'])
def getSeed():
    seed = 0

    return seed




if("__name__" == "__main__"):
    application.run("0.0.0.0")