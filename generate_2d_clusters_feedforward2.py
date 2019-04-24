#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py

import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
# Hyper-parameters 
image_size = 30
input_size = image_size*image_size
hidden_size = 500
num_classes = 2 ### 0,1
num_epochs = 10
batch_size = 100
learning_rate = 0.001



def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list
    
def genetate_percolation_system2d(size=40,prob = 0.4):    
    rnd_arr  = np.random.rand(size,size)
    mark_arr = np.zeros((size, size)).astype(int)

    #print(mark_arr)

    for x in range(size):
        for y in range(size):
            if rnd_arr[x][y]<prob:
                rnd_arr[x][y]=1.0
            else:
                rnd_arr[x][y]=0.0
    #print(rnd_arr)


    cluster_cnt=0

    def mark(x,y,cluster_num,recursion_level):
        if recursion_level>3000:
            return
        mark_arr[x][y]=cluster_num
        for dx in[-1,0,1]:
            for dy in[-1,0,1]:
                xx=x+dx
                yy=y+dy
                if xx<size and yy<size and xx>=0 and yy>=0:
                    if mark_arr[xx][yy]==0 and rnd_arr[xx][yy]==1:
                        mark(xx,yy,cluster_num,recursion_level+1)
    #			else:
    #				print(xx,yy)

        
    cluster_cnt=0
    for x in range(size):
        for y in range(size):
            if rnd_arr[x][y]==1 and mark_arr[x][y]==0:
                cluster_cnt=cluster_cnt+1
    #			print((x,y,cluster_cnt))
                mark(x,y,cluster_cnt,0)

    #print(mark_arr)

    perc_list = []
    for x in range(size):
        if mark_arr[x][0]>0:
            cluster_id = mark_arr[x][0]
            for xx in range(size):
                if mark_arr[xx][size-1]==cluster_id and not(cluster_id in perc_list):
                    perc_list.append(cluster_id)

    if False:
        for x in range(size):
            s=""
            for y in range(size):
                m=mark_arr[x][y]
                if m==0:
                    s=s+"  "
                else:
                    sm=str(m).ljust(2)
                    if m in perc_list:
                        s=s+"~ "
                    else:
                        s=s+sm
            print(s)
    return [len(perc_list),flatten(rnd_arr)]

#    print("percolation clusters: ")
#    print(perc_list)

#print(genetate_percolation_system2d(10,0.4))


############################################################ nn #############################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        leak=0.01
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 =  nn.ReLU() # nn.LeakyReLU(leak, inplace=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.relu2 =  nn.ReLU() #nn.LeakyReLU(leak, inplace=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.relu3 =   nn.ReLU() #nn.LeakyReLU(leak, inplace=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 =   nn.ReLU() #nn.LeakyReLU(leak, inplace=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 =   nn.ReLU() #nn.LeakyReLU(leak, inplace=True)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.relu6 =   nn.ReLU() #nn.LeakyReLU(leak, inplace=True)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.fc7(out)
        out = self.fc_out(out)
        return out


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size)
            ,nn.Sigmoid()
            ,nn.Linear(hidden_size, num_classes)
            ,nn.Sigmoid() #нужно ли ? или criterion вместо этого?
        ).to(device)

        #def myrand(m):
        #return  np.random.uniform(0, np.sqrt(2/100))
        #    return 0
        #self.net.apply(myrand)

    def forward(self, x):
        return self.net(x)

model = NeuralNet2(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def strRatio(c1, c2):
    if (c2 > 0):
        return "{:.2f};({}/{})".format(c1 / c2, c1, c2)
    else:
        return "{:.2f};({}/{})".format(0.0, c1, c2)

def test_model(prob):
    with torch.no_grad():
        correct = 0
        total = 0
        perc_count=0
        arr_labels=[]
        arr_images=[]
        for i in range(batch_size):
            arr = genetate_percolation_system2d(image_size,prob)
            isPercolation = arr[0]
            img = arr[1]
            if isPercolation>0:
                perc_count+=1
            arr_labels.append(isPercolation)
            arr_images.append(img)

        images = torch.tensor(arr_images,  dtype=torch.float, device=device)
        labels = torch.tensor(arr_labels,  dtype=torch.long, device=device)
        outputs = model(images)
        
        if len(labels)!=len(outputs):
            print("ERR len(labels)!=len(outputs)")

        total=0
        total0=0
        total1=0
        
        correct=0
        correct0=0
        correct1=0
        #print(labels)
        #print(outputs)
        for i in range(len(labels)):
            l=int(labels[i])
            o=outputs[i]
            total+=1
            
            if l==0:
                total0+=1
            if l>0:
                total1+=1
            
            if o[0]>o[1]:
                predicted_l=0
            else:
                predicted_l=1
                
            if predicted_l==l:
                correct +=1
                
            if predicted_l==0 and l==0:
                correct0 +=1
                
            if predicted_l==1 and l>0:
                correct1 +=1

        print("prob="+"{:.2f}".format(prob)+" perc="+strRatio(perc_count,batch_size)+";prediction="+strRatio(correct,total)+";0prediction="+strRatio(correct0, total0 )+";1prediction="+strRatio(correct1, total1 ) )

            
              
        #_, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()



        #print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

#test_model()
# Train the model
start_millis = (round(time.time() * 10))/10

#total_step = len(train_loader)
for iprob in range(35 , 45 , 1):
    prob=iprob*0.01
    for epoch in range(num_epochs):
        arr_labels=[]
        arr_images=[]

        perc_count=0
        batch=[]
        for i in range(batch_size):
            arr = genetate_percolation_system2d(image_size,prob)
            isPercolation = arr[0]
            img = arr[1]
            if isPercolation>0:
                perc_count+=1
            arr_labels.append(isPercolation)
            arr_images.append(img)
            batch.append(arr)

        #print("perc_count = {:.2f}".format(perc_count/batch_size))

        for i in range(batch_size):

            # Move tensors to the configured device
            images = torch.tensor(arr_images,  dtype=torch.float, device=device)
            labels = torch.tensor(arr_labels,  dtype=torch.long, device=device)
            #images = images.reshape(-1, image_size*image_size).to(device)
            #labels = labels.to(device)

            # Forward pass
            if False:
                print('labels:')
                print(labels)
            #print(images)
            #print(len(images))

            outputs = model(images)
            if False:
                print('outputs:')
                print(outputs)
                #exit()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_millis = ( round( time.time() * 10 ) )/10
            time_delta = current_millis - start_millis

            if False and (i < 2 or i>batch_size-2):
                print ('v1 Epoch [{}/{}], Step [{}], Loss: {:.4f} time={:.1f}'
                       .format(epoch+1, num_epochs, i+1,  loss.item(),time_delta))


    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
 #   print("Test the model")

    test_model(prob)

#torch.save(model.state_dict(), 'model.ckpt')

