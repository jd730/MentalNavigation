import numpy as np
from .data_utils import read_pkl
from scipy import stats
import matplotlib.pyplot as plt
from .theory_utils import nCr



def plot_erormat(err_gcpc, lambdas, Npatts):
    print(err_gcpc.shape)
    #plt.figure(figsize=(5, 10))
    #plt.figure()
    #plt.imshow(np.mean(err_gcpc[:,:], axis=2), interpolation='nearest', aspect='auto')  #avg error over 100 trials
    #m = np.mean(err_gcpc[:,:], axis=2)
    #plt.plot(m[8])
    plt.imshow(err_gcpc[:,:,0], interpolation='nearest', aspect='auto')
    #plt.plot(err_gcpc[18,:,0]*Npatts)
    plt.colorbar()
    plt.ylabel("number of place cells")
    plt.xlabel("number of patterns")
    plt.title(f"Cleanup Error (single trial), lambdas={lambdas}")


    # plt.figure(2)
    # plt.plot()
    plt.show()
    exit()


def process_data(filename, results_dir, errthresh=0.02, error="err_gcpc"):
  fname = f"{results_dir}/{filename}" 

  data = read_pkl(fname)
  err_gcpc = data[error]
  Np_lst = data["Np_lst"]
  nruns = data["nruns"]
  Npos = data["Npos"]
  Ng = data["Ng"]
  lambdas = data["lambdas"]

  Npatts = np.arange(1,Npos+1)
  #print(err_gcpc.shape)
  # plot_erormat(err_gcpc, lambdas, Npatts)
  # exit()

  capacity = -1*np.ones((len(Np_lst), nruns))
  valid = err_gcpc <= errthresh   # bool
  for Np in range(len(Np_lst)):
    # Original conservative
    # for r in range(nruns):
    #   lst = np.argwhere(valid[Np,:,r] == False)
    #   #lst = np.argwhere(valid[Np,:] == False)
    #   if len(lst) == 0:
    #     #print("full capacity")
    #     capacity[Np,r] = Npos
    #   else:      
    #     capacity[Np,r] = lst[0]-1

    # # relaxed capacity
    for r in range(nruns):
      lst = np.argwhere(valid[Np,:,r] == True)
      # print(valid[Np,:,r] == True)
      # print(len(lst))
      if len(lst) == 0:
        capacity[Np,r] = 0
      else: 
        capacity[Np,r] = lst[-1]    
   #    capacity[Np,r] = lst[-1] 
  avg_cap = np.mean(capacity, axis=1)     # mean error over runs
  #std_cap = np.std(capacity, axis=1)     # std dev over runs
  std_cap = stats.sem(capacity, axis=1)   # std error of the mean
  #avg_cap = avg_cap/nCr(Ng,3) #np.prod(lambdas)
  #std_cap = std_cap/nCr(Ng,3) #np.prod(lambdas) #nCr(Ng,3) #
  return avg_cap, std_cap, Np_lst, Ng, lambdas



# # # Capacity based on mean error over runs
# def process_data(filename, results_dir, errthresh=0.02, error="err_gcpc"):
#   fname = f"{results_dir}/{filename}" 

#   data = read_pkl(fname)
#   err_gcpc = data[error]
#   Np_lst = data["Np_lst"]
#   nruns = data["nruns"]
#   Npos = data["Npos"]
#   Ng = data["Ng"]
#   lambdas = data["lambdas"]

#   err_gcpc = np.mean(err_gcpc, axis=2)    
    
#   #plot_erormat(err_gcpc, lambdas)
#   errthresh = errthresh    # 1% error
#   capacity = -1*np.ones((len(Np_lst)))
#   valid = err_gcpc <= errthresh   # bool
#   for Np in range(len(Np_lst)):
#     # Original conservative
#    #  lst = np.argwhere(valid[Np,:] == False)
#    #  if len(lst) == 0:
#       # #print("full capacity")
#    #      capacity[Np] = Npos
#    #  else:      
#    #      capacity[Np] = lst[0]-1

#     # # relaxed capacity
#     lst = np.argwhere(valid[Np,:] == True)
#     capacity[Np] = lst[-1]  
#   avg_cap = capacity 
#   std_cap = 0
#   #std_cap = stats.sem(capacity, axis=1)   # std error of the mean

#   return avg_cap, std_cap, Np_lst, Ng, lambdas


# def process_contdata():
#   data = read_pkl(f"{results_dir}/{filename}")
#   err_gcpc = data['err_gcpc']
#   Np_lst = data['Np_lst']
#   Npos = data['Npos']
#   period_factor = data['period_factor']

