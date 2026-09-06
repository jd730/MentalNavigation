from src.assoc_utils_np import *
from numpy.random import shuffle 
from .data_utils import *
from .assoc_utils_np import *


# # nearest neighbour
def cleanup(s, sbook):
    sclean = np.zeros_like(s)
    for i in range(len(sclean)):  #runs
        idx = np.argmax(s[i].T@sbook)
        sclean[i,:,0] = sbook[:,idx]
    return sclean


# nearest neighbour
# def cleanup(s, sbook):
#   a,b,c = s.shape
#   sr = np.reshape(s, (a,c,b))
#   idx = np.argmax(sr@sbook, axis=2)
#   sclean = sbook[:,idx]
#   x,y,z = sclean.shape
#   sclean = np.reshape(sclean, (y,x,z))
#   return sclean


def train_sensory(pbook, sbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', sbook[:,:Npatts], pbook[:,:,:Npatts]) 


def dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns):
    s = sinit
    p = np.sign(Wps@s)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g)
        # if i%2 == 0:
        #   p = np.sign(Wpg@g) 
        # else:
        #   p = np.sign(Wps@s) 
    #s = topk_binary(sin, sparsity)
    sin = Wsp@p
    s = np.sign(sin)    
    scup = cleanup(s, sbook)
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*Ns) #(2*sparsity);
    errsenscup = np.sum(np.abs(scup-strue), axis=(1,2))/(2*Ns)
    return errpc, errgc, errsens, errsenscup 

# dynamics_gs() but instead of returning errors, just return the p, g, s, scup patterns
def dynamics_gs_raw(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns):
    s = sinit
    p = np.sign(Wps@s)
    for i in range(Niter):
        gin = Wgp@p
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g)
        # if i%2 == 0:
        #   p = np.sign(Wpg@g) 
        # else:
        #   p = np.sign(Wps@s) 
    #s = topk_binary(sin, sparsity)
    sin = Wsp@p
    s = np.sign(sin)    
    scup = cleanup(s, sbook)
    return p, g, s, scup

def dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    g = gtrue
    p = np.sign(Wpg@g)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        if i%2 == 0:
            p = np.sign(Wpg@g) 
        else:
            p = np.sign(Wps@s)      
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net 
        if i%2 == 0:
            p = np.sign(Wpg@g) 
        else:
            p = np.sign(Wps@s) 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens    



def senstrans_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        print("k=",k)
        #Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights
        #Wsp=np.zeros((nruns, Ns,Np));           # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)


        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            srep = np.zeros((nruns, *strue.shape))
            srep[:,:,:] = strue  #(nruns,Ns,1)
            sinit = corrupt_p(Ns, pflip, srep, nruns)      # make corrupted sc pattern
            errpc, errgc, errsens, errsenscup = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns) 


            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
            sum_senscup += errsenscup   
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        err_senscup[k,:] = sum_senscup/Npatts
        k += 1   
    return err_pc, err_gc, err_sens, err_senscup  

# senstrans_gs() but instead of returning average errors, 
# return the p, g, s, scup patterns for each run/input pattern
# plus Wpg, pbook setup
def senstrans_gs_raw(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_train, 
    Npatts_test, nruns, Ns, sbook, sparsity, incremental=False):
    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    # Learning patterns 
    if incremental:
        Wgp = train_gcpc(pbook, gbook, Npatts_test)
    else:
        Wgp = train_gcpc(pbook, gbook, Npatts_train)
    #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)
    Wsp = pseudotrain_Wsp(sbook, pbook, Npatts_train)
    Wps = pseudotrain_Wps(pbook, sbook, Npatts_train)

    # store outputs (p, g, s, scup)
    outputs = [np.zeros((nruns, dim, Npatts_test)) for dim in [Np, Ng, Ns, Ns]]

    # Testing
    for x in range(Npatts_test): 
        ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
        gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
        strue = sbook[:,x,None]       # true (noiseless) sensory pattern
        
        srep = np.zeros((nruns, *strue.shape))
        srep[:,:,:] = strue  #(nruns,Ns,1)
        sinit = corrupt_p(Ns, pflip, srep, nruns)      # make corrupted sc pattern
        results = dynamics_gs_raw(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                            Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns) 
        p, g, s, scup = results
        for i in range(4):
            outputs[i][:, :, x] = results[i].squeeze()
        
        if incremental and x >= Npatts_train:
            if x == Npatts_train:
                sbook_obs = np.repeat(sbook[None, :, :x], nruns, axis=0)
                pbook_obs = pbook[:, :, :x].copy()
            sbook_obs = np.concatenate((sbook_obs, s), axis=2)
            pbook_obs = np.concatenate((pbook_obs, p), axis=2)
            n = x + 1 # since x is 0-indexed
            Wsp = pseudotrain_3d(sbook_obs, pbook_obs, n)
            Wps = pseudotrain_3d(pbook_obs, sbook_obs, n)

    return outputs, Wpg, pbook

# different implementation of incremental: does incremental Wgp as well
def senstrans_gs_raw_v2(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_train, 
    Npatts_test, nruns, Ns, sbook, sparsity, incremental=False):
    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    # Learning patterns 
    Wgp = train_gcpc(pbook, gbook, Npatts_train)
    #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)
    Wsp = pseudotrain_Wsp(sbook, pbook, Npatts_train)
    Wps = pseudotrain_Wps(pbook, sbook, Npatts_train)

    # store outputs (p, g, s, scup)
    outputs = [np.zeros((nruns, dim, Npatts_test)) for dim in [Np, Ng, Ns, Ns]]

    # Testing
    for x in range(Npatts_test): 
        ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
        gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
        strue = sbook[:,x,None]       # true (noiseless) sensory pattern
        
        srep = np.zeros((nruns, *strue.shape))
        srep[:,:,:] = strue  #(nruns,Ns,1)
        sinit = corrupt_p(Ns, pflip, srep, nruns)      # make corrupted sc pattern
        results = dynamics_gs_raw(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                            Wsp, Wps, sparsity, gtrue, sinit, strue, Np, sbook, Ns) 
        p, g, s, scup = results
        for i in range(4):
            outputs[i][:, :, x] = results[i].squeeze()
        
        if incremental and x >= Npatts_train:
            if x == Npatts_train:
                sbook_obs = np.repeat(sbook[None, :, :x], nruns, axis=0)
                pbook_obs = pbook[:, :, :x].copy()
                gbook_obs = np.repeat(gbook[None, :, :x], nruns, axis=0)
            sbook_obs = np.concatenate((sbook_obs, s), axis=2)
            pbook_obs = np.concatenate((pbook_obs, p), axis=2)
            gbook_obs = np.concatenate((gbook_obs, g), axis=2)
            n = x + 1 # since x is 0-indexed
            Wgp = train_gcpc_3d(pbook_obs, gbook_obs, n)
            Wsp = pseudotrain_3d(sbook_obs, pbook_obs, n)
            Wps = pseudotrain_3d(pbook_obs, sbook_obs, n)

    return outputs, Wpg, pbook

def senstrans_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        #Wsp = train_sensory(pbook, sbook, Npatts)
        #Wps = np.transpose(Wsp, axes=(0,2,1))
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)
    

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]     # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens      


def senstrans_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) #/ (np.sqrt(M))                       # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))   # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        #Wsp = train_sensory(pbook, sbook, Npatts)
        #Wps = np.transpose(Wsp, axes=(0,2,1))
        Wsp = pseudotrain_Wsp(sbook, pbook, Npatts)
        Wps = pseudotrain_Wps(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern

            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            errpc, errgc, errsens = dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wps, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens
