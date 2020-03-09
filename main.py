from VolumeRatioUtils import VolumeRatio
import json

if __name__=='__main__':
    with open('params.json') as f:
        params = json.load(f)
    
    test_accuracy_fn =  '%s_test_accuracy_%s.npy' % (params['dataset'], params['RUNID'])
    samples_fn = '%s_samples_%s.npy' % (params['dataset'], params['RUNID'])
    #initialize sampler
    sampler = VolumeRatio(dataset=params['dataset'], N_start=params['N_start'], N_finish=params['N_finish'], N_by=params['N_by'],
                         n_cores=params['n_cores'], label1=params['label1'], label2=params['label2'], corrupt_fraction=params['corrupt_fraction'], 
                         fourier_features=params['fourier_features']) 
    #generate samples
    sampler.sample(n_samples=params['n_samples'], save_fn=samples_fn) 
    #compute test accuracies
    sampler.compute_test_accuracies(m=params['n_test_points'], save_fn=test_accuracy_fn) 