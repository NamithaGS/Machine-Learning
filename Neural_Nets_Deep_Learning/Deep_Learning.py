import hw_utils as utils


if __name__ == "__main__":
    X_tr_,y_tr_,X_te_,y_te_ = utils.loaddata("MiniBooNE_PID.txt")
    X_tr, X_te = utils.normalize(X_tr_, X_te_)
    y_tr, y_te = utils.normalize(y_tr_, y_te_)
    din = X_tr.shape[1] # Size of the Input features
    dout = y_tr.shape[1] #Size of the Output predictions


    #d. Linear Activation
    print "\nd. Linear Activation."
    firstarch = [[din, dout], [din, 50, dout], [din, 50, 50, dout], [din, 50, 50, 50, dout]]
    print "     First Architecture :"
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, firstarch, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
    print "\n       Second Architecture :"
    secondarch = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout],[din, 800, 800, 500, 300, dout]]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, secondarch, actfn='linear', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)

    #e. Sigmoid Activation
    print "\ne. Sigmoid Activation"
    sigmoidarch = [[din, 50, dout],[din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, sigmoidarch, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)

    #f. Relu Activation
    print "\nf. Relu Activation"
    reluarch = sigmoidarch
    best, times =utils.testmodels(X_tr, y_tr, X_te, y_te, reluarch, actfn='relu', last_act='softmax', reg_coeffs=[0.0],
                num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)

    #g. L-2 Regularization
    print "\ng. L-2 Regularization"
    l2arch = [[din, 800, 500, 300, dout]]
    l2regparam = [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)] ## [ 1e-7 , 5e-7 , 1e-6 , 5e-6 ,1e-5]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, l2arch, actfn='relu', last_act='softmax', reg_coeffs=l2regparam,
                num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
    best_lambda_noEstop = best[1]
    print "Best Regularization hyperparameter=", best_lambda_noEstop

    #h. Early Stopping and L2 regularization
    print "\nh. Early Stopping and L2 regularization"
    l2arch = [[din, 800, 500, 300, dout]]
    ##l2regparam = [ 1e-7 , 5e-7 , 1e-6 , 5e-6 ,1e-5]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, l2arch, actfn='relu', last_act='softmax', reg_coeffs=l2regparam,
                num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=True, verbose=0)
    best_lambda_Estop = best[1]
    best_lambda_Estop_acc = best[5]
    print("Best Regularization hyperparameter=", best_lambda_Estop)

    # if (best_lambda_noEstop_acc >  best_lambda_Estop_acc ):
    #     best_l2reg = best_lambda_noEstop
    # else:
    #     best_l2reg = best_lambda_Estop
    best_l2reg = best_lambda_Estop

    #i. SGD with weight decay
    print "\ni. SGD with weight decay"
    sgdarch =[ [din, 800, 500, 300, dout]]
    l2regparam = [5e-7]
    sgddecay= [pow(10,-5), 5*pow(10,-5), pow(10,-4), 3*pow(10,-4), 7*pow(10,-4), pow(10,-3)]##[ 1e-5 , 5e-5 , 1e-4 , 3e-4 ,7e-4, 1e-3]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, sgdarch, actfn='relu', last_act='softmax', reg_coeffs=l2regparam,
                num_epoch=100, batch_size=1000, sgd_lr=5e-5, sgd_decays=sgddecay, sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
    bestdecay = best[2]
    print "Best Decay=", bestdecay

    #bestdecay =[pow(10,-5), 5*pow(10,-5), pow(10,-4), 3*pow(10,-4), 7*pow(10,-4), pow(10,-3)]##[ 1e-5 , 5e-5 , 1e-4 , 3e-4 ,7e-4, 1e-3]
    #best_l2reg= [pow(10,-7), 5*pow(10,-7), pow(10, -6), 5*pow(10, -6), pow(10, -5)] ## [ 1e-7 , 5e-7 , 1e-6 , 5e-6 ,1e-5]

    #j. Momentum
    print "\nj. Momentum"
    sgdarch = [[din, 800, 500, 300, dout]]
    l2regparam = [0.0]
    moms = [0.99, 0.98, 0.95, 0.9,0.85]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, sgdarch, actfn='relu', last_act='softmax', reg_coeffs=l2regparam,
                num_epoch=50, batch_size=1000, sgd_lr=1e-5, sgd_decays=[bestdecay], sgd_moms=moms,
                    sgd_Nesterov=True, EStop=False, verbose=0)
    best_mom = best[3]
    print "Best Momentum=", best_mom


    #k. Combining the Above
    print "\nk. Combining the Above"
    sgdarch = [[din, 800, 500, 300, dout]]
    l2regparam = [0.0]
    moms = [0.99, 0.98, 0.95, 0.9,0.85]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, sgdarch, actfn='relu', last_act='softmax', reg_coeffs=[best_l2reg],
                num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[bestdecay], sgd_moms=[best_mom],
                    sgd_Nesterov=True, EStop=True, verbose=0)
    best_acc = best[5]
    print "Best Accuracy=", best_acc

    #l. Grid Search with Validation
    print "\nl. Grid Search with Validation"
    gridsearcharch = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout], [din, 800, 500, 300, dout], [din, 800, 800,500, 300, dout]]
    l2regparam = [ 1e-7 , 5e-7 , 1e-6 , 5e-6 ,1e-5]
    decays= [pow(10,-5), 5*pow(10,-5), pow(10,-4)]##[ 1e-5 , 5e-5 , 1e-4]
    best, times = utils.testmodels(X_tr, y_tr, X_te, y_te, gridsearcharch, actfn='relu', last_act='softmax', reg_coeffs=l2regparam,
                num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=decays, sgd_moms=[0.99],
                    sgd_Nesterov=True, EStop=True, verbose=0)
    best_acc = best[5]
    print "Best Accuracy=", best_acc