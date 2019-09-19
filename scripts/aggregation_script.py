#!/usr/bin/python3

import os
import numpy as np
import tensorflow as tf
from progress.bar import Bar

root = "/cs-share/dream/RSS-Kingfisher/training_script/results"
exp = ["ASCTEC","DRONE","HERON"]
#net_arch = ["CNN", "MLP","RNN","LSTM","GRU"]

print("Running preliminary checks...")
for exp_name in exp:
    print("  Checking "+exp_name+" files")
    net_arch = os.listdir(os.path.join(root,exp_name))
    for tech_name in net_arch:
        print("    Checking "+str(tech_name)+" architectures")
        sub_root = os.path.join(root,exp_name,tech_name)
        directory_list = os.listdir(sub_root)
        exp_list = [x[:-3] for x in directory_list]
        exp_list = list(dict.fromkeys(exp_list))
        exp_nb = len(exp_list)
        print("      "+str(exp_nb)+" experiments found")
        success = 0
        for arch in exp_list:
            i = 0
            for i in range(5):
                if len(os.listdir(os.path.join(sub_root,arch+'-r'+str(i)))) > 4:
                    i+= 1
            if i == 5:
                success+=1
            else:
                print("      + Experience "+arch+" failed.")
        print("      Architectures success rate: "+str(np.round(success/exp_nb,3)*100))
print()
r = input("  Check done ! Do you want to continue ? [y/n] ")
if r == "r":
    print("    Exiting...")
    exit(0)
elif r =="y":
    pass
else:
    print("Unknown request shuting down...")
    exit(0)

# We want to quantify how each experience faired in average have a rough idea of
# the standard deviation during training. The results will be expressed for
# each variables and in average. Them self expressed as a function of a
# multistep prediction or as a function of single step predictions. The time to
# reach the best accuracy will also be stored.
print(" ")
print("Aggregating results...")
#import tensorflow as tf

results = {}
try:
  os.mkdir(os.path.join(root,"aggregation"))
except:
    pass

for exp_name in exp:
    print("  Aggregating "+exp_name+" files")
    net_arch = os.listdir(os.path.join(root,exp_name))
    for tech_name in net_arch:
        results[exp_name] = {}
        table = [["architecture_name","min_train_acc","var_train_acc","min_val_ss_mean_accuracy","min_val_ms_mean_accuracy","time_to_min_ss","time_to_min_ms","execution_time","parameters_number","memory"]]
        print("    Aggregating "+str(tech_name)+" architectures")
        sub_root = os.path.join(root,exp_name,tech_name)
        directory_list = os.listdir(sub_root)
        exp_list = [x[:-3] for x in directory_list]
        exp_list = list(dict.fromkeys(exp_list))
        exp_nb = len(exp_list)
        bar = Bar('      Processing', max=exp_nb)
        best_value = 1000000
        best_name = ""
        best_ms_value = 1000000
        best_ms_name = ""
        for arch in exp_list:
            avg_tr_acc = []
            avg_tr_var = []
            avg_vl_ss_acc = []
            avg_vl_ms_acc = []
            ttss = []
            ttms = []
            exe = []
            param = []
            mem = []
            tti = []
            for i in range(5):
                subsub_root = os.path.join(sub_root,arch+'-r'+str(i))
                if i == 0:
                    tf.reset_default_graph()
                    try:
                        saver = tf.train.import_meta_graph(os.path.join(subsub_root,"final_NN.meta"))
                    except:
                        saver = tf.train.import_meta_graph(os.path.join(subsub_root,"final.meta"))
                    #for op in tf.get_default_graph().get_operations():
                        #print(str(op.name))
                    #builder = tf.profiler.ProfileOptionBuilder
                    #opts = builder(builder.time_and_memory()).order_by('micros').build()
                    
                    #pctx = tf.contrib.tfprof.ProfileContext('./train_dir',trace_steps=[], dump_steps=[])
                    total_parameters = 0
                    #print(arch)
                    for variable in tf.trainable_variables():
                        # shape is an array of tf.Dimension
                        shape = variable.get_shape()
                        #print(shape)
                        #print(len(shape))
                        variable_parameters = 1
                        for dim in shape:
                            #print(dim)
                            variable_parameters *= dim.value
                            #print(variable_parameters)
                        total_parameters += variable_parameters
                    
                    #print(total_parameters)
                    #with tf.Session() as sess:
                        #saver.restore(sess, os.path.join(subsub_root,"final_NN"))
                        #profiler = tf.profiler.Profiler(sess.graph)
                        #run_meta = tf.RunMetadata()
                        #opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                        #pctx.trace_next_step()
                        #pctx.dump_next_step()
                        #sess.run('output/BiasAdd:0',feed_dict={'inputs:0':np.zeros([1000,156])},options = opt,run_metadata=run_meta)
                        #if exp_name == "ASCTEC":
                        #    input_dim = 13
                        #elif exp_name == "DRONE":
                        #    input_dim = 8
                        #elif exp_name == "HERON":
                        #    input_dim = 5
                        #try:
                        #    sess.run('output/BiasAdd:0',feed_dict={'inputs:0':np.zeros([16,input_dim*12])})
                        #except:
                        #    sess.run('output_1/BiasAdd:0',feed_dict={'inputs:0':np.zeros([16,input_dim*12])})
                        #profiler.add_step(0, run_meta)
                        #option_builder = tf.profiler#.ProfileOptionBuilder
                        #profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder.trainable_variables_parameter()))
                        #opts = option_builder.ProfileOptionBuilder.time_and_memory()
                        #profiler.profile_operations(options=opts)

                        #pctx.profiler.profile_operations(options=opts)
                        #print('haaa')
                        #sess.close()

                tr_loss_log = np.load(os.path.join(subsub_root,"train_loss_log.npy"))
                ts_ss_log = np.load(os.path.join(subsub_root,"test_single_step_loss_log.npy"))
                ts_ms_log = np.load(os.path.join(subsub_root,"test_multi_step_loss_log.npy"))
                #print(np.mean(tr_loss_log[:,2:],axis=-1).shape)
                #exit(0)
                avg_tr_acc.append(np.min(np.mean(tr_loss_log[:,2],axis=-1)))
                avg_tr_var.append(np.var(np.mean(tr_loss_log[:,2],axis=-1)))
                #avg_vl_ss_acc.append(np.min(np.mean(ts_ss_log[:,2],axis=-1)))
                ind = np.argmin(np.mean(ts_ss_log[:,2:],axis=-1))
                inf_time = ts_ss_log[ind,1]/ind
                avg_vl_ss_acc.append(np.mean(ts_ss_log[ind,2:]))
                ttss.append(ts_ss_log[ind,1])
                #avg_vl_ms_acc.append(np.min(ts_ms_log[:,2]))
                ind = np.argmin(ts_ms_log[:,2])
                avg_vl_ms_acc.append(ts_ms_log[ind,2])
                ttms.append(ts_ms_log[ind,1])
                tti.append(inf_time)
               

                #pttf_gh_ss = os.path.join(subsub_root,"best_1S_NN")
                #pttf_mt_ss = os.path.join(subsub_root,"best_1S_NN.meta")
                #pttf_gh_ms = os.path.join(subsub_root,"best_NN")
                #pttf_mt_ms = os.path.join(subsub_root,"best_NN.meta")  
            avg_tr_acc = np.mean(avg_tr_acc)
            avg_tr_var = np.mean(avg_tr_var)
            avg_vl_ss_acc = np.mean(avg_vl_ss_acc)
            avg_vl_ms_acc = np.mean(avg_vl_ms_acc)
            ttss = np.mean(ttss)
            ttms = np.mean(ttms)
            tti = np.mean(tti)
            line = [arch, str(avg_tr_acc), str(avg_tr_var), str(avg_vl_ss_acc),
                    str(avg_vl_ms_acc), str(ttss), str(ttms), str(tti), str(total_parameters),str(0)]
            table.append(line)
            
            if best_value > avg_vl_ss_acc:
                best_value = avg_vl_ss_acc
                best_name = arch
            if best_ms_value > avg_vl_ms_acc:
                best_ms_value = avg_vl_ms_acc
                best_ms_name = arch
            bar.next()
        with open(os.path.join(root,"aggregation/results-"+exp_name+"-"+tech_name+".csv"),"a") as filename:
            table = [",".join(x)+"\n" for x in table]
            filename.write("".join(table))
        print(" ")
        print("      Best single architecture: "+best_name+" with accuracy:"+str(best_value))
        print("      Best multistep architecture: "+best_ms_name+" with accuracy:"+str(best_ms_value))
print("  Aggregation Done !")
print(" ")
print("Performing quick analysis")

