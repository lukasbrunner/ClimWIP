#!/usr/bin/python
'''
File Name : func_calc_wu_wq.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 07-03-2016
Modified: Fri 21 Oct 2016 14:37:35 CEST
Purpose: calculate wu, weight accounting for dependency between models
         in multi-model ensemble, and wq, weigth accounting for
         quality compared to observations, and approximation based on weights

'''
import numpy as np

def calc_wu(delta_u, model_names, sigmaS2 = 0.6):
    if (type(sigmaS2) is np.ndarray):
        wu = np.empty((len(sigmaS2), len(model_names), len(model_names)))
        wu.fill(np.NaN)
        for s in xrange(len(sigmaS2)):
            S = np.exp( - ((delta_u / sigmaS2[s]) ** 2))
            for j in xrange(len(model_names)):
                for jj in xrange(len(model_names)):
                    S_tmp = np.copy(S[jj, :]) # temporary copy
                    S_tmp[jj] = 0
                    S_tmp[j] = 0
                    Ru = 1 + (np.sum(S_tmp))
                    wu[s, j, jj] = 1 / Ru

                    del(S_tmp)
                    del(Ru)
    else:
        wu = np.empty((len(model_names)))
        wu.fill(np.NaN)
        S = np.exp( - ((delta_u / sigmaS2) ** 2))
        for j in xrange(len(model_names)):
            S_tmp = np.copy(S[j, :]) # temporary copy
            S_tmp[j] = 0
            Ru = 1 + (np.sum(S_tmp))
            wu[j] = 1 / Ru
            del(S_tmp)
            del(Ru)
    return wu

def calc_wq(delta_q, model_names, sigmaD2 = 0.6):
    if (type(sigmaD2) is np.ndarray):
        wq = np.empty((len(sigmaD2), len(model_names), len(model_names)))
        wq.fill(np.NaN)
        for s in xrange(len(sigmaD2)):
            for j in xrange(len(model_names)):
                for jj in xrange(len(model_names)):
                    wq[s, j, jj] = np.exp( - ((delta_q[jj, j] / sigmaD2[s]) ** 2))
    else:
        wq = np.empty((len(model_names)))
        wq.fill(np.NaN)
        for j in xrange(len(model_names)):
            wq[j] =  np.exp( - ((delta_q[j] / sigmaD2) ** 2))

    return wq

def calc_weights_approx(wu, wq, model_names, data, var_file = None):
    if (type(data) == dict()):
        if model_names != data.keys():
            print('Warning: models in list and dict not the same')
    if (len(wq) == len(model_names)):
        w_prod = np.empty((len(wq)))
        w_prod.fill(np.NaN)
        weights = {}
        for j in xrange(len(model_names)):
            tmp_wu = wu[j]
            w_prod[j] = wq[j] * tmp_wu

        wu_wq_sum = np.nansum(w_prod)
        for j in xrange(len(model_names)):
            ref = model_names[j]
            if wu_wq_sum != 0.0:
                weights[ref] =  w_prod[j] / wu_wq_sum
            else:
                weights[ref] = w_prod[j] * 0.0
        tmp = 0
        for key, value in data.iteritems():
            if (var_file == 'std'):
                tmp_pow = weights[key] * np.power(value, 2)
                tmp = tmp + tmp_pow
            else:
                tmp = tmp + weights[key] * value
        if (var_file == 'std'):
            approx = np.sqrt(tmp / np.nansum(weights.values()))
        else:
            approx = tmp / np.nansum(weights.values())
    else:
        weights = np.empty((len(wu), len(wq), len(model_names),
                            len(model_names)))
        weights.fill(np.NaN)
        data_array = np.array(data, dtype = float)
        dims = data_array.shape
        if len(dims) == 3 :
            approx = np.empty((len(wu), len(wq), len(model_names), dims[1],
                               dims[2], dims[3]))
        elif len(dims) == 2:
            approx = np.empty((len(wu), len(wq), len(model_names), dims[1]))
        else:
            approx = np.empty((len(wu), len(wq), len(model_names)))
        approx.fill(np.NaN)
        for u in xrange(len(wu)):
            for q in xrange(len(wq)):
                for j in xrange(len(model_names)):
                    # model_names_out.append(model_names[j])
                    # set model to compare to to zero, perfect model approach
                    tmp_wu = wu[u, j, :]
                    tmp_wu[j] = 0.0
                    # set model to compare to to zero, perfect model approach
                    tmp_wq = wq[q, j, :]
                    tmp_wq[j] = 0.0
                    w_prod = tmp_wq * tmp_wu

                    wu_wq_sum = np.nansum(w_prod)
                    if wu_wq_sum != 0:
                        tmpw =  w_prod / wu_wq_sum
                    else:
                        tmpw =  w_prod * 0
                    if len(dims) == 3 :
                        tmpw_resh = np.reshape(np.repeat(tmpw,
                                               dims[1] * dims[2] * dims[3]),
                                               (len(model_names), dims[1],
                                                dims[2], dims[3]))
                        if (var_file == 'std'):
                            tmp = (tmpw_resh * np.power(data_array, 2))
                            approx[u, q, j, :, :, :] = np.sqrt(np.nansum
                                                               (tmp,
                                                                axis = 0))
                        else:
                            tmp = (tmpw_resh * data_array)
                            approx[u, q, j, :, :, :] = np.nansum(tmp, axis = 0)
                    elif len(dims) == 2:
                        tmpw_resh = np.reshape(np.repeat(tmpw, dims[1]),
                                               (len(model_names), dims[1]))

                        if (var_file == 'std'):
                            tmp = (tmpw_resh * np.power(data_array, 2))
                            approx[u, q, j, :] = np.sqrt(np.nansum
                                                         (tmp, axis = 0))
                        else:
                            tmp = (tmpw_resh * data_array)
                            approx[u, q, j, :] = np.nansum(tmp, axis = 0)
                    else:
                        if (var_file == 'std'):
                            tmp = (tmpw * np.power(data_array, 2))
                            approx[u, q, j] = np.sqrt(np.nansum
                                                      (tmp, axis = 0))
                        else:
                            tmp = (tmpw * data_array)
                            approx[u, q, j] = np.nansum(tmp, axis = 0)
                    del tmp
                    weights[u, q, j, :] = tmpw

    return {'weights':weights, 'approx':approx}
