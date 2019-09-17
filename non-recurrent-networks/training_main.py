
import os
import argparse

#main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dataset',nargs='+',type=str, default='..', help='path to the data file to be used')
    parser.add_argument('--output', type=str, default='./result', help='path to the folder to save file to')
    parser.add_argument('--split_test', type=int, default='0', help='split test/train+val set for  cross-validation')
    parser.add_argument('--split_val', type=int, default='0', help='split train/val set for  cross-validation')
    parser.add_argument('--batch_size', type=int, default='64', help='size of the batch')
    parser.add_argument('--super_batch_size', type=int, default='512', help='size of the super batch')
    parser.add_argument('--iterations', type=int, default='10000', help='number of iterations')
    parser.add_argument('--used_points', type=int, default='12', help='number of points used to predict the outputs')
    parser.add_argument('--pred_points', type=int, default='1', help='number of point to predict')
    parser.add_argument('--frequency', type=int, default='1', help='sampling frequency')
    parser.add_argument('--test_window', type=int, default='30', help='multi step test window size')
    parser.add_argument('--test_iter', type=int, default='5', help='multi step test iteration')
    parser.add_argument('--alpha', type=float, default='0.4', help='PER param alpha')
    parser.add_argument('--beta', type=float, default='0.5', help='PER param beta')
    parser.add_argument('--experience_replay', type=int, default='4', help='PER replay times')
    parser.add_argument('--model_type', type=str, default='classic', help='type of network, one of : classic, loss')
    parser.add_argument('--ds_type', type=str, default='boat', help='type of dataset, one of : boat, asctec, drone')
    parser.add_argument('--optimisation_type', type=str, default='none', help='type of optimisation, one of : none, grad, PER')
    parser.add_argument('--input_state_dim', type=int, default='5', help='input state dimension')
    parser.add_argument('--output_state_dim', type=int, default='3', help='output state dimention')
    parser.add_argument('--cmd_dim', type=int, default='2', help='command dimention')

    args = parser.parse_args()

    # CHECK IF FINAL WEIGTHS FILES HAVE BEEN GENERATED
    # IF SO LINE ALREADY DONE / IF NOT GOOD TO DO
    # --- CODE RAN USING PARALLEL SSH ---

    #if someone else already did the job, stopping there otherwise we do it
    if not os.path.exists(args.output+"/final_NN.index"):
        import neuralnetworks
        NN = neuralnetworks.NN(args)
        NN.train()


