## Training a Recommender System in PySpark



def parse_data(line):
    try:
        line_array = line.split(',')
        return (line_array[6],line_array[1]) # user-term pairs
    except:
        return None




f = open('Online Retail.csv',encoding="Windows-1252")
purchases = []
users = {}
items = {}
user_index = 0
item_index = 0

for index, line in enumerate(f):
    if index > 0: # skip header
        purchase = parse_data(line)
        if purchase is not None:
            if users.get(purchase[0],None) is not None:
                purchase_user = users.get(purchase[0])
            else:
                users[purchase[0]] = user_index
                user_index += 1
                purchase_user = users.get(purchase[0])
            if items.get(purchase[1],None) is not None:
                purchase_item = items.get(purchase[1])
            else:
                items[purchase[1]] = item_index
                item_index += 1
                purchase_item = items.get(purchase[1])
                purchases.append((purchase_user,purchase_item))>>>f.close()

    
    
purchasesRdd = sc.parallelize(purchases,5).map(lambda x: Rating(x[0],x[1],1.0))



from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

k = 10
iterations = 10
mfModel = ALS.train(purchasesRdd, k, iterations)