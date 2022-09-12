import os
import numpy as np
import json
import math
import time
import sys
import ast

from scipy import stats

from ..dl.optimizers import *
from ..dl.loss_functions import *
from ..dl.activation_functions import *
from ..dl.layers import *

from ..globals import g
from ..utils import get_key, dump_data, get_ftp_credentials, load_data, image_to_column, column_to_image, output_shape, pooling_layer_output_shape 
from ..ftp import get_client as get_ftp_client
from ..ftp import check_credentials as check_credentials
from ..config import FTP_DOWNLOAD_FILES_FOLDER
from ..strings import functions



numpy_functions = {
            "neg": "np.negative",
            "pos": "np.positive",
            "add": "np.add",
            "sub": "np.subtract",
            "exp": "np.exp",
            "natlog": "np.log",
            "square":"np.square",
            "pow":"np.power",
            "square_root":"np.sqrt",
            "cube_root":"np.cbrt",
            "abs":"np.abs", 
            "sum":"sum", 
            "sort":"np.sort",
            "reverse":"np.flip",
            "min":"np.min",
            "max":"max",
            "argmax":"np.argmax",
            "argmin":"np.argmin",
            "transpose":"np.transpose",
            "div":"np.divide",
            'mul': 'np.multiply',
            'matmul': 'np.matmul',
            'multiply':'np.multiply',
            'dot': 'np.dot',
            'split': 'np.split', 
            'reshape':'reshape', 
            'unique': 'np.unique', 
            'expand_dims':'expand_dims', 
            'inv': 'np.linalg.inv', 
            'gather': 'gather', 
            'stack': 'np.stack', 
            'tile': 'np.tile', 
            'slice': 'ravslice',

            'find_indices': 'find_indices',
            'shape':'shape',
            'squeeze':'np.squeeze',
            'pad':'pad',
            'index':'index',

            #Comparision ops
            'greater': 'np.greater',
            'greater_equal':'np.greater_equal' ,
            'less': 'np.less',
            'less_equal':'np.less_equal' ,
            'equal':'np.equal' ,
            'not_equal': 'np.not_equal',
            'logical_and':'np.logical_and' ,
            'logical_or': 'np.logical_or',
            'logical_not': 'np.logical_not',
            'logical_xor': 'np.logical_xor',

            #statistics
            'mean': 'mean',
            'average': 'np.average',
            'mode': 'mode',
            'variance': 'variance',
            'std': 'np.std', 
            'percentile': 'np.percentile',
            'random': 'np.random',
            'bincount': 'np.bincount',
            'where': 'where',
            #'sign': Operators.SIGN,  
            'foreach': 'foreach',
            'set_value': 'set_value',
            'clip': 'clip',
            'random_uniform': 'np.random.uniform',
            'prod': 'np.prod',
            'flatten': 'flatten',
            'ravel': 'np.ravel',

            'concat': 'concatenate',
            'cube': 'np.cbrt',
            'arange':'np.arange',
            'repeat':'repeat',
            'join_to_list': 'join_to_list',
            'combine_to_list': 'combine_to_list',
            'zeros':'np.zeros',
            'ravint':'ravint',
            'cnn_index':'cnn_index',
            'cnn_add_at':'cnn_add_at',
            'cnn_index_2':'cnn_index_2',
            'size': 'size',

            # Machine Learning Ops
            'linear_regression': 'linear_regression',
            'logistic_regression': 'logistic_regression',
            'knn_classifier': 'knn_classifier',
            'knn_regressor': 'knn_regressor',
            'naive_bayes': 'naive_bayes',
            'kmeans': 'kmeans',
            'svm_svc': 'svm_svc',
            'svm_svr': 'svm_svr',
            'decision_tree_classifier': 'decision_tree_classifier',
            'decision_tree_regressor': 'decision_tree_regressor',
            'random_forest_classifier': 'random_forest_classifier',
            'random_forest_regressor': 'random_forest_regressor',
            
            'forward_pass_dense': 'forward_pass_dense',
            'backward_pass_dense': 'backward_pass_dense',

            'forward_pass_batchnorm': 'forward_pass_batchnorm',
            'backward_pass_batchnorm': 'backward_pass_batchnorm',

            'forward_pass_dropout': 'forward_pass_dropout',
            'backward_pass_dropout': 'backward_pass_dropout',

            'forward_pass_activation': 'forward_pass_activation',
            'backward_pass_activation': 'backward_pass_activation',

            'forward_pass_conv2d': 'forward_pass_conv2d',
            'backward_pass_conv2d': 'backward_pass_conv2d',

            'forward_pass_flatten': 'forward_pass_flatten',
            'backward_pass_flatten': 'backward_pass_flatten',

            'forward_pass_maxpool2d': 'forward_pass_maxpool2d',
            'backward_pass_maxpool2d': 'backward_pass_maxpool2d',


            'square_loss': 'square_loss',
            'square_loss_gradient': 'square_loss_gradient',
            'cross_entropy_loss': 'cross_entropy_loss',
            'cross_entropy_gradient': 'cross_entropy_gradient',
            'cross_entropy_accuracy': 'cross_entropy_accuracy',

    }

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'selu': SELU,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    'softplus': SoftPlus
}

def compute_locally_bm(*args, **kwargs):
    operator = kwargs.get("operator", None)
    op_type = kwargs.get("op_type", None)
    param_args =kwargs.get("params",None)
    # print("Operator", operator,"Op Type:",op_type)
    if op_type == "unary":
        value1 = args[0]
        t1=time.time()
        params=""
        if param_args is not None:
            params=[op_param_mapping[operator][str(_)] for _ in param_args ]
        # print(params)
        expression="{}({})".format(numpy_functions[operator], value1['value'])
        eval(expression)
        t2=time.time()
        return t2-t1

    elif op_type == "binary":
        value1 = args[0]['value']
        value2 = args[1]['value']
        t1=time.time()
        eval("{}({},{})".format(numpy_functions[operator], value1, value2))
        t2=time.time()
        return t2-t1

# async 
def compute_locally(payload, subgraph_id, graph_id):
    try:
        # print("Computing ",payload["operator"])
        # print('\n\nPAYLOAD: ',payload)

        values = []


        for i in range(len(payload["values"])):
            if "value" in payload["values"][i].keys():
                # print("From server")
                if "path" not in payload["values"][i].keys():
                    values.append(payload["values"][i]["value"])

                else:
                    server_file_path = payload["values"][i]["path"]

                    download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER,os.path.basename(payload["values"][i]["path"]))

                    # try:
                    g.ftp_client.download(download_path, os.path.basename(server_file_path))
                    value = load_data(download_path).tolist()
                    # print('Loaded Data Value: ',value)
                    values.append(value)

                    # except Exception as error:
                    #     print('Error: ', error)
                    #     emit_error(payload, error, subgraph_id, graph_id)

                    if os.path.basename(server_file_path) not in g.delete_files_list and payload["values"][i]["to_delete"] == 'True':
                        g.delete_files_list.append(os.path.basename(server_file_path))

                    if os.path.exists(download_path):
                        os.remove(download_path)

            elif "op_id" in payload["values"][i].keys():
                # print("From client")
                # try:
                values.append(g.outputs[payload['values'][i]['op_id']])
                # except Exception as e:
                #     emit_error(payload,e, subgraph_id, graph_id)

        payload["values"] = values

        # print("Payload Values: ", payload)

        op_type = payload["op_type"]
        operator = payload["operator"]
        params=payload['params']
        param_string=""
        for i in params.keys():
            if type(params[i]) == str:
                temp = ast.literal_eval(params[i])
                if type(temp) == dict:
                    param_string+=","+i+"="+str(params[i])
                else:    
                    param_string+=","+i+"=\'"+str(params[i])+"\'"
            elif type(params[i]) == dict and 'op_id' in params[i].keys():
                op_id = params[i]["op_id"]
                param_value = g.outputs[op_id]
                if type(param_value) == str:
                    param_string+=","+i+"=\'"+str(param_value)+"\'"
                else:
                    param_string+=","+i+"="+str(param_value)
            else:
                param_string+=","+i+"="+str(params[i])


        # try:
        if op_type == "unary":
            value1 = payload["values"][0]
            short_name = get_key(operator,functions)
            result = eval("{}({}{})".format(numpy_functions[short_name], value1,param_string))

        elif op_type == "binary":
            value1 = payload["values"][0]
            value2 = payload["values"][1]
            short_name = get_key(operator,functions)
            expression="{}({}, {}{})".format(numpy_functions[short_name], value1, value2,param_string)



            result = eval(expression)

        if 'sklearn' in str(type(result)):
            file_path = upload_result(payload, result)

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                # 'username': g.cid,
                # 'token': g.ravenverse_token,
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

        if 'dict' in str(type(result)):
            file_path = upload_result(payload, result)

            g.outputs[payload["op_id"]] = result

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                # 'username': g.cid,
                # 'token': g.ravenverse_token,
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })


        if not isinstance(result, np.ndarray):
            result = np.array(result)

        result_byte_size = result.size * result.itemsize

        if result_byte_size < (30 * 1000000)//10000:
            try:
                result = result.tolist()
            except:
                result = result

            g.outputs[payload["op_id"]] = result

            return json.dumps({
            'op_type': payload["op_type"],
            'result': result,
            # 'username': g.cid,
            # 'token': g.ravenverse_token,
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "success"
            })

        else:

            file_path = upload_result(payload, result)

            g.outputs[payload["op_id"]] = result.tolist()

            # op = g.ops[payload["op_id"]]
            # op["status"] = "success"
            # op["endTime"] = int(time.time() * 1000)
            # g.ops[payload["op_id"]] = op

            return json.dumps({
                'op_type': payload["op_type"],
                'file_name': os.path.basename(file_path),
                # 'username': g.cid,
                # 'token': g.ravenverse_token,
                'operator': payload["operator"],
                "op_id": payload["op_id"],
                "status": "success"
            })

    except Exception as error:
        print('Error: ', error)
        if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
            print('\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
            sys.exit()

        emit_error(payload, error, subgraph_id, graph_id)


def upload_result(payload, result):
    # result_size = result.size * result.itemsize
    try:
        result = result.tolist()
    except:
        result=result
    
    # print("Emit Success")

    file_path = dump_data(payload['op_id'],result)
    g.ftp_client.upload(file_path, os.path.basename(file_path))
    
    # print("\nFile uploaded!", file_path)#, ' Size: ', result_size)
    os.remove(file_path)
  
    return file_path
    

def emit_error(payload, error, subgraph_id, graph_id):
    print("Emit Error")
    # print(payload)
    # print(error)
    g.error = True
    error=str(error)
    client = g.client
    print(error,payload)
    client.emit("op_completed", json.dumps({
            'op_type': payload["op_type"],
            'error': error,
            'operator': payload["operator"],
            "op_id": payload["op_id"],
            "status": "failure",
            "subgraph_id": subgraph_id,
            "graph_id": graph_id
    }), namespace="/client")

    # op = g.ops[payload["op_id"]]
    # op["status"] = "failure"
    # op["endTime"] = int(time.time() * 1000)
    # g.ops[payload["op_id"]] = op

    try:
        for ftp_file in g.delete_files_list:
            g.ftp_client.delete_file(ftp_file)
    except Exception as e:

        g.delete_files_list = []
        g.outputs = {}
        g.has_subgraph = False

    g.delete_files_list = []
    g.outputs = {}
    g.has_subgraph = False


#ops:
def ravslice(tensor,begin=None,size=None):
    result=tensor[begin:begin+size]
    return result

def gather(tensor,indices):
    result=[]
    for i in indices:
        result.append(tensor[i])
    return result
    
def where(a,b,condition=None):
    if condition is None:
        raise Exception("condition is missing")
    else:
        result=np.where(condition,a,b)
    return result

def clip(a,lower_limit=None,upper_limit=None):
    if lower_limit is None:
        raise Exception("lower limit is missing")
    elif upper_limit is None:
        raise Exception("upper limit is missing")
    else:
        result = np.clip(a,lower_limit,upper_limit)
    return result

def max(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    if isinstance(axis,list):
        axis= tuple(axis)
    result=np.max(a,axis=axis,keepdims=keepdims)
    return result

def mean(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False

    if isinstance(axis,list):
        axis= tuple(axis)
    result=np.mean(a,axis=axis,keepdims=keepdims)
    return result

def variance(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False

    if isinstance(axis,list):
        axis= tuple(axis)
    result=np.var(a,axis=axis,keepdims=keepdims)
    return result

def sum(a,axis=None,keepdims=False):
    if str(keepdims) == 'True':
        keepdims = True
    else:
        keepdims = False
    if isinstance(axis,list):
        axis= tuple(axis)
    result=np.sum(a,axis=axis,keepdims=keepdims)
    return result

def flatten(a):
    a = np.array(a)
    return a.flatten()

def split(arr,numOrSizeSplits=None,axis=None):
    result=np.split(arr,numOrSizeSplits,axis=axis)
    return result

def expand_dims(arr,**kwargs):
    axis=kwargs.get('axis')
    if axis is not None:
        result=np.expand_dims(arr,axis=axis)
    else:
        result= np.expand_dims(arr,axis=0)
    return result

def tile(arr,reps):
    if reps is not None:
        result=np.tile(arr,reps)
    else:
        emit_error()
    return result

def one_hot_encoding(arr,depth):
    return np.squeeze(np.eye(depth)[arr.reshape(-1)])

def foreach(val=None,**kwargs):
    operator=kwargs.get("operation")
    result=[]
    paramstr=""
    del kwargs['operation']
    print(kwargs)
    for _ in kwargs.keys():
        paramstr+=","+_+"="+str(kwargs.get(_))
    for i in val:
        evalexp="{}({}{})".format(numpy_functions[operator],i,paramstr)
        print("\n\nevaluating:",evalexp)
        res=eval(evalexp)
        if type(res) is np.ndarray:
            result.append(res.tolist())
        else:
            result.append(res)
    return result

def find_indices(arr,val):
    result=[]
    
    for i in val:
        indices = [_ for _, arr in enumerate(arr) if arr == i]
        result.append(indices)
    if len(val) == 1:
        return indices
    else:
        return result

def reshape(tens,shape=None):
    if shape is None:
        return None
    else:
        return np.reshape(tens,newshape=shape)

def mode(arr,axis=0):
    result=stats.mode(arr,axis=axis)
    return result.mode

def concatenate(*args,**kwargs):
    param_string=""
    for i in kwargs.keys():
        if type(params[i]) == str:
            param_string+=","+i+"=\'"+str(params[i])+"\'"
        else:
            param_string+=","+i+"="+str(params[i])
    result=eval("np.concatenate(args"+param_string+")")
    return result

def shape(arr,index=None):
    arr = np.array(arr)
    if index is None:
        return arr.shape
    else:
        return arr.shape[int(index)]

def pad(arr,sequence=None,mode=None):
    if sequence is None:
        raise Exception("sequence param is missing")
    elif mode is None:
        raise Exception("mode is missing")
    arr = np.array(arr)
    result = np.pad(arr,sequence,mode=mode)
    return result

def repeat(arr,repeats=None, axis=None):
    if repeats is None:
        raise Exception("repeats param is missing")

    arr = np.array(arr)
    result = np.repeat(arr,repeats=repeats,axis=axis)
    return result

def index(arr,indices=None):
    if indices is None:
        raise Exception("indices param is missing")
    if isinstance(indices, str):
        # arr = np.array(arr)
        result = eval("np.array(arr)"+indices)
    else:
        result = eval("np.array(arr)[{}]".format(tuple(indices)))
    return result

def join_to_list(a,b):
    a = np.array(a)
    result = np.append(a,b)
    return result

def combine_to_list(a,b):    
    result = np.array([a,b])
    return result

def ravint(a):
    return int(a)

def cnn_index(arr,index1=None,index2=None,index3=None):
    if index1 is None or index2 is None or index3 is None:
        raise Exception("index1, index2 or index3 param is missing")
    
    result = eval("np.array(arr)"+"[:,{},{},{}]".format(index1,index2,index3))
    return result

def cnn_index_2(a, pad_h=None, height=None, pad_w=None, width=None):
    if pad_h is None or height is None or pad_w is None or width is None:
        raise Exception("index1, index2 or index3 param is missing")

    a = np.array(a)
    result = a[:, :, pad_h:height+pad_h, pad_w:width+pad_w]
    return result

def cnn_add_at(a, b, index1=None,index2=None,index3=None):
    if index1 is None or index2 is None or index3 is None:
        raise Exception("index1, index2 or index3 param is missing")
    
    a = np.array(a)
    b = np.array(b)
    index1 = np.array(index1)
    index2 = np.array(index2)
    index3 = np.array(index3)

    np.add.at(a, (slice(None), index1, index2, index3), b)
    return a

def set_value(a,b,indices):
    if indices is None:
        raise Exception("indices param is missing")
    if isinstance(indices, str):
        exec("a"+indices+'='+'b')
    else:
        print("\n\n Indices in set value: ", indices)
        a = np.array(a)
        a[tuple(indices)] = b
    return a

def size(a):
    a = np.array(a)
    return a.size

def linear_regression(x,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x, y) 
    return model

def knn_classifier(x,y,k=None):
    if k is None:
        raise Exception("k param is missing")
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k).fit(x, y) 
    return model

def knn_regressor(x,y,k=None):
    if k is None:
        raise Exception("k param is missing")
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k).fit(x, y) 
    return model

def logistic_regression(x, y, random_state=0):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=random_state).fit(x, y) 
    return model

def naive_bayes(x, y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB().fit(x, y) 
    return model

def kmeans(x, n_clusters=None):
    if n_clusters is None:
        raise Exception("n_clusters param is missing")
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters).fit(x) 
    return model

def svm_svc(x, y, kernel='linear'):
    from sklearn.svm import SVC
    model = SVC(kernel=kernel).fit(x, y) 
    return model

def svm_svr(x, y, kernel='linear'):
    from sklearn.svm import SVR
    model = SVR(kernel=kernel).fit(x, y) 
    return model

def decision_tree_classifier(x, y):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier().fit(x, y) 
    return model

def decision_tree_regressor(x, y):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor().fit(x, y) 
    return model

def random_forest_classifier(x, y, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split).fit(x, y) 
    return model

def random_forest_regressor(x, y, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split).fit(x, y) 
    return model

def square_loss(y_true, y_pred):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return SquareLoss().loss(y_true, y_pred)

def square_loss_gradient(y, y_pred):
    y_pred = np.array(y_pred['result'])
    y = np.array(y)
    return SquareLoss().gradient(y, y_pred)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return CrossEntropy().loss(y_true, y_pred)

def cross_entropy_gradient(y, y_pred):
    y_pred = np.array(y_pred['result'])
    y = np.array(y)
    return CrossEntropy().gradient(y, y_pred)

def cross_entropy_accuracy(y_true, y_pred):
    y_pred = np.array(y_pred['result'])
    y_true = np.array(y_true)
    return CrossEntropy().acc(y_true, y_pred)


def forward_pass_dense(X, n_units=None, input_shape=None, data=None, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])
    
    if data is None: 
        limit = 1 / math.sqrt(input_shape[0])
        W  = np.random.uniform(-limit, limit, (input_shape[0], n_units))
        w0 = np.zeros((1, n_units))
        W_opt_state_dict = None
        w0_opt_state_dict = None
    else:
        W = np.array(data['W'])
        w0 = np.array(data['w0'])
        W_opt_state_dict = data['W_opt_state_dict']
        w0_opt_state_dict = data['w0_opt_state_dict']
    
    result = X.dot(W)+w0
    forward_pass_output = {
        'W': W.tolist(),
        'w0': w0.tolist(),
        'result': result.tolist(),
        'W_opt_state_dict': W_opt_state_dict,
        'w0_opt_state_dict': w0_opt_state_dict
    }
    return forward_pass_output

def backward_pass_dense(accum_grad, layer_input=None, optimizer=None,data=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    W_init = np.array(data['W'])
    W = np.array(data['W'])
    w0 = np.array(data['w0'])
        
    # Calculate gradient w.r.t layer weights
    grad_w = layer_input.T.dot(accum_grad)
    grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

    optimizer_data = optimizer
    optimizer_name = optimizer_data['name']
    del optimizer_data['name']
    
    if optimizer_name == "RMSprop": 
        if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
            W_opt = RMSprop(**optimizer_data)
            w0_opt = RMSprop(**optimizer_data)
        else:
            W_opt = RMSprop(**optimizer_data, **data['W_opt_state_dict'])
            w0_opt = RMSprop(**optimizer_data, **data['w0_opt_state_dict'])
    
    if optimizer_name == "Adam": 
        if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
            W_opt = Adam(**optimizer_data)
            w0_opt = Adam(**optimizer_data)
        else:
            W_opt = Adam(**optimizer_data, **data['W_opt_state_dict'])
            w0_opt = Adam(**optimizer_data, **data['w0_opt_state_dict'])

    # Update the layer weights
    W = W_opt.update(W, grad_w)
    w0 = w0_opt.update(w0, grad_w0)

    # Return accumulated gradient for next layer
    # Calculated based on the weights used during the forward pass

    accum_grad = accum_grad.dot(W_init.T)

    backward_pass_output = {
        'W': W.tolist(),
        'w0': w0.tolist(),
        'accum_grad': accum_grad.tolist(),
        'W_opt_state_dict': W_opt.state_dict(),
        'w0_opt_state_dict': w0_opt.state_dict()
    }
    return backward_pass_output

def forward_pass_batchnorm(X, input_shape=None, momentum=None, eps=None, training="True", trainable="True", data=None, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if data is None:    
        if len(input_shape) == 1:
            shape = (1, input_shape[0])
        else:
            shape = (1, input_shape[0], 1, 1)

        running_mean = np.zeros(shape)
        running_var = np.ones(shape)
        gamma = np.ones(shape)
        beta = np.zeros(shape)
        gamma_opt_state_dict = None
        beta_opt_state_dict = None
    else:
        running_mean = np.array(data['running_mean'])
        running_var = np.array(data['running_var'])
        gamma = np.array(data['gamma'])
        beta = np.array(data['beta'])
        gamma_opt_state_dict = data['gamma_opt_state_dict']
        beta_opt_state_dict = data['beta_opt_state_dict']

    if training == "True" and trainable == "True":
        if len(input_shape) == 1:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
        else:
            mean = np.mean(X, axis=(0,2,3), keepdims=True)
            var = np.var(X, axis=(0,2,3), keepdims=True)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        mean = running_mean
        var = running_var

    # Statistics saved for backward pass
    X_centered = X - mean
    stddev_inv = 1 / np.sqrt(var + eps)

    X_norm = X_centered * stddev_inv
    output = gamma * X_norm + beta

    forward_pass_output = {
        'running_mean': running_mean.tolist(),
        'running_var': running_var.tolist(),
        'X_centered': X_centered.tolist(),
        'stddev_inv': stddev_inv.tolist(),
        'gamma': gamma.tolist(),
        'beta': beta.tolist(),
        'result': output.tolist(),
        'gamma_opt_state_dict': gamma_opt_state_dict,
        'beta_opt_state_dict': beta_opt_state_dict
    }
    return forward_pass_output

def backward_pass_batchnorm(accum_grad, input_shape=None, optimizer=None, trainable="True", data=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    # Save parameters used during the forward pass
    gamma_init = np.array(data['gamma'])
    gamma = np.array(data['gamma'])
    beta = np.array(data['beta'])
    X_centered = np.array(data['X_centered'])
    stddev_inv = np.array(data['stddev_inv'])

    # If the layer is trainable the parameters are updated
    if trainable=="True":
        X_norm = X_centered * stddev_inv
        if len(input_shape) == 1:
            grad_gamma = np.sum(accum_grad * X_norm, axis=0, keepdims=True)
            grad_beta = np.sum(accum_grad, axis=0, keepdims=True)
        else:
            grad_gamma = np.sum(accum_grad * X_norm, axis=(0,2,3), keepdims=True)
            grad_beta = np.sum(accum_grad, axis=(0,2,3), keepdims=True)

        optimizer_data = optimizer
        optimizer_name = optimizer_data['name']
        del optimizer_data['name']
        
        if optimizer_name == "RMSprop":
            if data['gamma_opt_state_dict'] is None and data['beta_opt_state_dict'] is None:
                gamma_opt = RMSprop(**optimizer_data)
                beta_opt = RMSprop(**optimizer_data)
            else:
                gamma_opt = RMSprop(**optimizer_data, **data['gamma_opt_state_dict'])
                beta_opt = RMSprop(**optimizer_data, **data['beta_opt_state_dict'])

        if optimizer_name == "Adam":
            if data['gamma_opt_state_dict'] is None and data['beta_opt_state_dict'] is None:
                gamma_opt = Adam(**optimizer_data)
                beta_opt = Adam(**optimizer_data)
            else:
                gamma_opt = Adam(**optimizer_data, **data['gamma_opt_state_dict'])
                beta_opt = Adam(**optimizer_data, **data['beta_opt_state_dict'])


        gamma = gamma_opt.update(gamma, grad_gamma)
        beta = beta_opt.update(beta, grad_beta)

    batch_size = accum_grad.shape[0]

    # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)

    if len(input_shape) == 1:
        accum_grad = (1 / batch_size) * gamma_init * stddev_inv * (batch_size * accum_grad - np.sum(accum_grad, axis=0,keepdims=True) 
                                                                            - X_centered * stddev_inv**2 * np.sum(accum_grad * X_centered, axis=0, keepdims=True))
    else:
        accum_grad = (1 / batch_size) * gamma_init * stddev_inv * (batch_size * accum_grad - np.sum(accum_grad, axis=(0,2,3),keepdims=True) 
                                                                            - X_centered * stddev_inv**2 * np.sum(accum_grad * X_centered, axis=(0,2,3), keepdims=True))

    backward_pass_output = {
        'gamma': gamma.tolist(),
        'beta': beta.tolist(),
        'running_mean': data['running_mean'],
        'running_var': data['running_var'],
        'accum_grad': accum_grad.tolist(),
        'gamma_opt_state_dict': gamma_opt.state_dict(),
        'beta_opt_state_dict': beta_opt.state_dict()
    }
    return backward_pass_output    
    
def forward_pass_dropout(X, p=None, training="True", input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if training == "True":
        _mask = np.random.uniform(size=X.shape) > p
        c = _mask * (1 / (1-p)) 
        output = X * c
        forward_pass_output = {
            '_mask': _mask.tolist(),
            'result': output.tolist()
        }
    else:
        output = X
        forward_pass_output = {
            'result': output.tolist()
        }
    
    return forward_pass_output

def backward_pass_dropout(accum_grad, data=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    _mask = np.array(data['_mask'])
    accum_grad = accum_grad * _mask

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output

def forward_pass_activation(X, act_data=None, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    activation_function = activation_functions[act_data['name']]()
    output = activation_function(X)

    forward_pass_output = {
        'result': output.tolist()
    }
    return forward_pass_output

def backward_pass_activation(accum_grad, layer_input=None, act_data=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    activation_function = activation_functions[act_data['name']]()
    accum_grad = activation_function.gradient(layer_input) * accum_grad

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output

def forward_pass_conv2d(X, input_shape=None, n_filters=None, filter_shape=None, stride=None, padding_data=None, data=None, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    if data is None: 
        filter_height, filter_width = filter_shape
        channels = input_shape[0]
        limit = 1 / math.sqrt(np.prod(filter_shape))
        W  = np.random.uniform(-limit, limit, size=(n_filters, channels, filter_height, filter_width))
        w0 = np.zeros((n_filters, 1))
        W_opt_state_dict = None
        w0_opt_state_dict = None
        
    else:
        W = np.array(data['W'])
        w0 = np.array(data['w0'])
        W_opt_state_dict = data['W_opt_state_dict']
        w0_opt_state_dict = data['w0_opt_state_dict']
    
    batch_size = X.shape[0]
    X_col = image_to_column(X, filter_shape, stride=stride, output_shape=padding_data['padding'])
    # Turn weights into column shape
    W_col = W.reshape((n_filters, -1))
    # Calculate output
    output = W_col.dot(X_col) + w0
    # Reshape into (n_filters, out_height, out_width, batch_size)
    output = output.reshape(output_shape(input_shape=input_shape, n_filters=n_filters, filter_shape=filter_shape, padding=padding_data['padding'], stride=stride) + (batch_size, ))
    # output = output.reshape(shape=(self.output_shape() + (batch_size, )))
    # Redistribute axises so that batch size comes first
    
    forward_pass_output = {
        'result': output.transpose(3,0,1,2).tolist(),
        'X_col': X_col.tolist(),
        'W_col': W_col.tolist(),
        'W': W.tolist(),
        'w0': w0.tolist(),
        'W_opt_state_dict': W_opt_state_dict,
        'w0_opt_state_dict': w0_opt_state_dict
    }
    
    return forward_pass_output

def backward_pass_conv2d(accum_grad, layer_input=None, n_filters=None, filter_shape=None, stride=None, padding_data=None, optimizer=None, data=None, trainable="True", input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    if isinstance(layer_input, dict):
        layer_input = np.array(layer_input['result'])
    else:
        layer_input = np.array(layer_input)

    X_col = np.array(data['X_col'])
    W_col = np.array(data['W_col'])
    W = np.array(data['W'])
    w0 = np.array(data['w0'])
    

    accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)

    if trainable=="True":
        grad_w = accum_grad.dot(X_col.T).reshape(W.shape)

        grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

        optimizer_data = optimizer
        optimizer_name = optimizer_data['name']
        del optimizer_data['name']
        
        if optimizer_name == "RMSprop":
            if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
                W_opt = RMSprop(**optimizer_data)
                w0_opt = RMSprop(**optimizer_data)
            else:
                W_opt = RMSprop(**optimizer_data, **data['W_opt_state_dict'])
                w0_opt = RMSprop(**optimizer_data, **data['w0_opt_state_dict'])

        if optimizer_name == "Adam":
            if data['W_opt_state_dict'] is None and data['w0_opt_state_dict'] is None:
                W_opt = Adam(**optimizer_data)
                w0_opt = Adam(**optimizer_data)
            else:
                W_opt = Adam(**optimizer_data, **data['W_opt_state_dict'])
                w0_opt = Adam(**optimizer_data, **data['w0_opt_state_dict'])

        # Update the layers weights
        W = W_opt.update(W, grad_w)
        w0 = w0_opt.update(w0, grad_w0)

    # Recalculate the gradient which will be propogated back to prev. layer
    accum_grad = W_col.T.dot(accum_grad)
    # Reshape from column shape to image shape
    accum_grad = column_to_image(accum_grad,
                                layer_input.shape,
                                filter_shape,
                                stride=stride,
                                output_shape=padding_data['padding'])

    backward_pass_output = {
        'accum_grad': accum_grad.tolist(),
        'W': W.tolist(),
        'w0': w0.tolist(),
        'W_opt_state_dict': W_opt.state_dict(),
        'w0_opt_state_dict': w0_opt.state_dict()
    }
    return backward_pass_output

def forward_pass_maxpool2d(X, input_shape=None, pool_shape=None, stride=None, padding_data=None, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    batch_size, channels, height, width = X.shape

    _, out_height, out_width = pooling_layer_output_shape(
        input_shape=input_shape, pool_shape=pool_shape, stride=stride
    )

    X = X.reshape(batch_size*channels, 1, height, width)
    X_col = image_to_column(X, pool_shape, stride, padding_data['padding'])

    # MaxPool specific method
    arg_max = np.argmax(X_col, axis=0).flatten()
    output = X_col[arg_max, range(arg_max.size)]
    cache = arg_max
    
    output = output.reshape(out_height, out_width, batch_size, channels)
    output = output.transpose(2, 3, 0, 1)

    forward_pass_output = {
        'result': output.tolist(),
        'X_col': X_col.tolist(),
        'cache': cache.tolist()
    }
    
    return forward_pass_output

def backward_pass_maxpool2d(accum_grad, input_shape=None, pool_shape=None, stride=None, padding_data=None, data=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:
        accum_grad = np.array(accum_grad['accum_grad'])

    cache = np.array(data['cache'])

    batch_size, _, _, _ = accum_grad.shape
    channels, height, width = input_shape
    accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()

    # MaxPool or AveragePool specific method

    accum_grad_col = np.zeros((np.prod(pool_shape), accum_grad.size))
    arg_max = cache
    accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad

    accum_grad = column_to_image(accum_grad_col, (batch_size * channels, 1, height, width), pool_shape, stride, padding_data['padding'])
    accum_grad = accum_grad.reshape((batch_size,) + tuple(input_shape))

    backward_pass_output = {
        'accum_grad': accum_grad.tolist()
    }
    return backward_pass_output



def forward_pass_flatten(X, input_layer=None):
    if input_layer == "True":
        X = np.array(X)
    else:
        X = np.array(X['result'])

    forward_pass_output = {
        'result': X.reshape((X.shape[0], -1)).tolist()
    }
    return forward_pass_output

def backward_pass_flatten(accum_grad, prev_input=None, input_layer=None):
    if input_layer == "True":
        accum_grad = np.array(accum_grad)
    else:    
        accum_grad = np.array(accum_grad['accum_grad'])
    
    if isinstance(prev_input, dict):
        prev_shape = np.array(prev_input['result']).shape
    else:
        prev_shape = np.array(prev_input).shape

    backward_pass_output = {
        'accum_grad': accum_grad.reshape(prev_shape).tolist()
    }
    return backward_pass_output
