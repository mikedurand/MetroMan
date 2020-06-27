from numpy import mean,disp,sqrt

def DispRMSEStats(Err,Truth,Prior,E):  

    #Quick Error stats
    Err.RelErrA0=mean(E.A0hat.T-Truth.A0)/mean(Truth.A0)

    disp(['Relative Error in A0:' +str(Err.RelErrA0)])
    disp(['Relative Uncertainty in A0: ' +str(mean(E.stdA0Post.T/E.A0hat.T))])


    RMSQPost=sqrt(mean( (E.QhatPostf.T-Truth.Q.T)**2,0 ))
    RMSQPrior=sqrt(mean( (E.QhatPrior.T-Truth.Q.T)**2,0 ))
    # ratio=((RMSQPrior-RMSQPost)/RMSQPrior)*100

    nR=len(E.A0hat)

    print('RMS for Q prior: ')
    for i in range(0,nR):
        print('%.1f' %RMSQPrior[i])


    print('RMS for Q posterior: ')
    for i in range(0,nR):
        print('%.1f' %RMSQPost[i])

    Err.QRelErrPrior=RMSQPrior/mean(Truth.Q.T,0)
    Err.QRelErrPost=RMSQPost/mean(Truth.Q.T,0)
 
    print('Average RMS for Q posterior: %.3f' %mean(Err.QRelErrPost,0))

    print('Average relative Q uncertainty: %.3f' %mean(mean(E.QstdPost/E.QhatPost,0)) )

    print('For entire timeseries, rRMSE= %.2f' %Err.Stats.rRMSE, 'and relative bias= %.2f' %Err.Stats.meanRelRes)
    
    RMSEQbart=Err.Stats.RMSE/Err.Stats.Qbart
    biasQbart=Err.Stats.bias/Err.Stats.Qbart
    print('For entire timeseries, RMSE/Qbart=%.2f' %RMSEQbart, \
          'and bias/Qbart=%.2f' %biasQbart)   

    return Err