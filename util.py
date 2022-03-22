import os, sys, pickle, math, argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

seed=400
np.random.seed(seed)

def exponential_decay_fn(epoch):
  return 0.05 * 0.1**(epoch / 20)

def lr_step_decay(epoch) : #initial_lr, drop, epoch_to_drop) :

    initial_lr = 0.01
    drop = 0.9
    epoch_to_drop = 3

    if epoch >= 50 :
        epoch == 50
    elif epoch >= 25 :
        epoch -= 25
#        new_lr = initial_lr
        print('INFO Setting learning rate back to initial LR (={})'.format(initial_lr))

    new_lr = np.round(initial_lr * math.pow(drop, math.floor((1+epoch)/epoch_to_drop)),4)
    print('INFO LR Schedule: {}'.format(new_lr))
    return new_lr

def Plot_Metrics(history, path_tosave):
    for x in range(0,len(history)):
        plt.plot(history[x].history['loss'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_loss'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

    saveit = "{}/{}".format(path_tosave, "dnn_lossepo.png")
    plt.savefig(saveit)
    plt.show()

    for x in range(0,len(history)):
        plt.plot(history[x].history['accuracy'], label='Train. fold-'+str(x))
        plt.plot(history[x].history['val_accuracy'], label='Val. fold-'+str(x))
        #plt.title('DNN ($D_{in}$=7, $D_{hidden}$=2, 0.2Drop)')
        plt.xlabel('epoch')
        plt.ylabel('accuracy in %')
        plt.legend(loc='lower right')

    saveit = "{}/{}".format(path_tosave, "dnn_accepo.png")
    plt.savefig(saveit)
    plt.show()

def mkdir_p(path) :
    import errno
    """
    Make a directory, if it exists silence the exception
    Args:
        path : full directory path to be made
    """
    try :
        os.makedirs(path)
    except OSError as exc :
        if exc.errno == errno.EEXIST and os.path.isdir(path) :
            pass
        else :
            raise

def make_nn_output_plots_oddeven( path_tosave, model = None, inputs = None, samples = None, targets = None, events=None) :

    #inputs_tek = inputs[events % 2==1]
    #inputs_cift = inputs[events % 2==0]

    ## Workaround if no eventnumbers are used:
    # Shuffle + split into two equal size arrays
    np.random.shuffle(inputs)
    inputs_tek = inputs_test[0:int(inputs.shape[0]/2),:]
    inputs_cift = inputs_test[int(inputs.shape[0]/2):int(inputs_test.shape[0]),:]

    nn_scores = np.ones([targets.size,2])
    nn_scores_tek = model[0].predict(inputs_tek,verbose = True)
    nn_scores_cift = model[1].predict(inputs_cift,verbose = True)
    nn_scores[0:int(inputs.shape[0]/2),0] = nn_scores_tek.flatten()
    nn_scores[0:int(inputs.shape[0]/2),1] = abs(nn_scores_tek - 1).flatten()
    nn_scores[int(inputs.shape[0]/2):int(inputs.shape[0]),0] = nn_scores_cift.flatten()
    nn_scores[int(inputs.shape[0]/2):int(inputs.shape[0]),1] = abs(nn_scores_cift - 1).flatten()
    class_labels = set(targets)
    targets_list = list(targets)
    nn_scores_dict = {}

    # index the sample names by their class label
    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    # break up the predicted scores by the class label
    for ilabel, label in enumerate(class_labels) :
        # left-most appearance of the label
        left = targets_list.index(label)
        # right-most appearance of the label
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        nn_scores_dict[label] = nn_scores[left:right+1]

    # start plotting
    for label in class_labels :
        #fig, ax = plt.subplots(1,1)
        fig = plt.figure()
        plt.grid(color='k', which='both', linestyle='--', lw=0.5, alpha=0.1, zorder = 0)
        plt.xlabel( "NN output for label {}".format(names[label]), horizontalalignment='right', x=1)
        #ax.set_xlim([1e-2,1.0])
        plt.xlim([0,1])
        #plt.yscale('log')
        histargs = {"bins":20, "range":(0,1.), "density":True, "histtype":'step'}
        #binning = np.arange(0,1,0.02)
        #centers = (binning[1:-2] + binning[2:-1])/2
        #ax.set_xlim((centers[0]-0.1, centers[-1]+0.1))
        for sample_label in nn_scores_dict :
            sample_scores_for_label = nn_scores_dict[sample_label][:]
            print(sample_scores_for_label)
            #sample_weights = sample_with_label(sample_label, samples).eventweights

            #yields, _ = np.histogram(sample_scores_for_label, bins = binning)
            plt.hist(sample_scores_for_label,label = names[sample_label], **histargs)
            #yields = yields/yields.sum()
            #ax.step(centers, yields[1:-1], label = names[sample_label], where = 'mid')

            #ax.hist(sample_scores_for_label, bins = binning, alpha = 0.3, label = names[sample_label], density = True)
        plt.legend(loc='best', frameon = False)
        savename = "nn_outputs_class_{}.pdf".format( names[label])

        savename = "{}/{}".format(path_tosave, savename)
        plt.savefig(savename, bbox_inches = 'tight', dpi = 200)
       # savename = "nn_outputs_{}_class_{}.pdf".format(path_tosave, names[label])

     #   savename = "{}/{}".format(path_tosave, savename)
     #   plt.savefig(savename, bbox_inches = 'tight', dpi = 200)

    return nn_scores

def make_nn_output_plots( model = None, inputs = None, samples = None, targets = None) :

    # set of scores for each label: shape = (n_samples, n_outputs)
    nn_scores = model.predict(inputs,verbose = True)

    class_labels = set(targets)
    targets_list = list(targets)
    nn_scores_dict = {}

    # index the sample names by their class label
    names = {}
    for sample in samples :
        names[sample.class_label()] = sample.name()

    # break up the predicted scores by the class label
    for ilabel, label in enumerate(class_labels) :
        # left-most appearance of the label
        left = targets_list.index(label)
        # right-most appearance of the label
        right = len(targets_list) - 1 - targets_list[::-1].index(label)
        nn_scores_dict[label] = nn_scores[left:right+1]

    # start plotting
    for label in class_labels :
        #fig, ax = plt.subplots(1,1)
        fig = plt.figure()
        plt.grid(color='k', which='both', linestyle='--', lw=0.5, alpha=0.1, zorder = 0)
        plt.xlabel( "NN output for label {}".format(names[label]), horizontalalignment='right', x=1)
        #ax.set_xlim([1e-2,1.0])
        plt.xlim([0,1])
        #plt.yscale('log')
        histargs = {"bins":20, "range":(0,1.), "density":True, "histtype":'step'}
        #binning = np.arange(0,1,0.02)
        #centers = (binning[1:-2] + binning[2:-1])/2
        #ax.set_xlim((centers[0]-0.1, centers[-1]+0.1))
        for sample_label in nn_scores_dict :
            sample_scores_for_label = nn_scores_dict[sample_label][:]
            print(sample_scores_for_label)
            #sample_weights = sample_with_label(sample_label, samples).eventweights

            #yields, _ = np.histogram(sample_scores_for_label, bins = binning)
            plt.hist(sample_scores_for_label,label = names[sample_label], **histargs)
            #yields = yields/yields.sum()
            #ax.step(centers, yields[1:-1], label = names[sample_label], where = 'mid')

            #ax.hist(sample_scores_for_label, bins = binning, alpha = 0.3, label = names[sample_label], density = True)
        plt.legend(loc='best', frameon = False)
        savename = "nn_outputs_class_{}.pdf".format( names[label])

        savename = "{}/{}".format(path_tosave, savename)
        plt.savefig(savename, bbox_inches = 'tight', dpi = 200)
       # savename = "nn_outputs_{}_class_{}.pdf".format(path_tosave, names[label])

     #   savename = "{}/{}".format(path_tosave, savename)
     #   plt.savefig(savename, bbox_inches = 'tight', dpi = 200)

    return nn_scores

def ScaleWeights(y,w):
    sum_wpos = sum( w[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg = sum( w[i] for i in range(len(y)) if y[i] == 0.0  )

    for i in range(len(w)):
        if (y[i]==1.0):
            w[i] = w[i] * (0.5/sum_wpos)
        else:
            w[i] = w[i] * (0.5/sum_wneg)

    w_av = sum(w)/len(w)
    w[:] = [x/w_av for x in w]

    sum_wpos_check = sum( w[i] for i in range(len(y)) if y[i] == 1.0  )
    sum_wneg_check = sum( w[i] for i in range(len(y)) if y[i] == 0.0  )

    print ('\n======Weight Statistic========================================')
    print ('Weights::        W(1)=%g, W(0)=%g' % (sum_wpos, sum_wneg))
    print ('Scaled weights:: W(1)=%g, W(0)=%g' % (sum_wpos_check, sum_wneg_check))
    print ('==============================================================')

def ScaleWeightsSignal(w,y):
    sumi = sum( w[i] for i in range(len(y)) if y[i] == 1.0 )


    for i in range(len(w)):
      if (y[i]==1.0):
        w[i] = sumi/len(w)
      else:
        w[i] = w[i]
