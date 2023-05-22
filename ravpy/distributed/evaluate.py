import json
import os
import sys
import socket
import ast
import pickle as pkl
from terminaltables import AsciiTable
import zipfile
from .compute import compute_locally, compute_backward, emit_error
from ..config import FTP_DOWNLOAD_FILES_FOLDER, FTP_TEMP_FILES_FOLDER
from ..globals import g
from ..utils import setTimeout, stopTimer, dump_data, dump_torch_model, dump_result_data
import time
import subprocess as sp
import torch
import time
import psutil
import gc

timeoutId = g.timeoutId
opTimeout = g.opTimeout
initialTimeout = g.initialTimeout

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    # memory_free_values = [psutil.virtual_memory()[3]/1000000000]
    return memory_free_values

@g.client.on('subgraph_forward', namespace="/client")
async def compute_subgraph_forward(d):
    g.param_queue['forward_param'] = d

@g.client.on('subgraph_backward', namespace="/client")
async def compute_subgraph_backward(d):
    g.param_queue['backward_param'] = d

async def compute_thread():
    while True:
        if g.param_queue.get('forward_param',None) is not None:
            await subgraph_forward_process(g.param_queue['forward_param'])
            del g.param_queue['forward_param']

        if g.param_queue.get('backward_param',None) is not None:
            await subgraph_backward_process(g.param_queue['backward_param'])
            del g.param_queue['backward_param']
        
        time.sleep(1.5)

async def subgraph_forward_process(d):
    gc.collect()
    total_t = time.time()
    g.logger.debug('subgraph_forward_process received')
    # os.system('clear')
    print(AsciiTable([['Provider Dashboard']]).table)
    g.dashboard_data.append([d["subgraph_id"], d["graph_id"], "Computing"])
    print(AsciiTable(g.dashboard_data).table)

    g.has_subgraph = True
    subgraph_id = d["subgraph_id"]
    graph_id = d["graph_id"]
    data = d["payloads"]
    # print("\n Data before computation: ", data)
    gpu_required = ast.literal_eval(d["gpu_required"])

    subgraph_outputs = d["subgraph_outputs_list"]
    persist_forward_pass_results_list = d["persist_forward_pass_results"]

    subgraph_zip_file_flag = d["subgraph_zip_file_flag"]
    results = []
    del_keys = []
    backward_called = False
    g.error = False
    # g.forward_computations = {}

    if subgraph_zip_file_flag == "True":
        server_file_path = 'zip_{}_{}.zip'.format(subgraph_id, graph_id)

        download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER, server_file_path)
        filesize = d['zip_file_size']
        try:
            g.ftp_client.ftp.voidcmd('NOOP')
            g.ftp_client.download(download_path, os.path.basename(server_file_path), filesize)

        except Exception as error:
            os.system('clear')
            g.dashboard_data[-1][2] = "Failed"
            print(AsciiTable([['Provider Dashboard']]).table)
            print(AsciiTable(g.dashboard_data).table)
            g.has_subgraph = False
            
            delete_dir = FTP_DOWNLOAD_FILES_FOLDER
            for f in os.listdir(delete_dir):
                os.remove(os.path.join(delete_dir, f))

            g.delete_files_list = []

            print('Error: ', error)
            emit_error(data[0], error, subgraph_id, graph_id)
            if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
                print(
                    '\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
                sys.exit()

        if os.path.exists(download_path):
            extract_to_path = FTP_DOWNLOAD_FILES_FOLDER
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_path)

            os.remove(download_path)
            g.ftp_client.delete_file(server_file_path)

    for index in data:
        if index['op_id'] in subgraph_outputs:
            to_upload = True
        else:
            to_upload = False
        operation_type = index["op_type"]
        operator = index["operator"]
        t1 = time.time()
        if operator == "start_backward_marker":
            marker_params = index["params"]
            step = ast.literal_eval(marker_params.get("step", "True"))
            for input_value in index["values"]:
                if "op_id" in input_value:
                    print("\n Backward Called")
                    g.forward_computations[input_value['op_id']].backward()
                    backward_called = True
                    if input_value['op_id'] in subgraph_outputs:
                        detached_loss = g.forward_computations[input_value['op_id']].detach()
                        g.forward_computations[input_value['op_id']] = detached_loss
                    else:
                        del g.forward_computations[input_value['op_id']]

            # for key in g.forward_computations.keys():
            #     if isinstance(g.forward_computations[key], dict):
            #         if g.forward_computations[key].get('result', None) is not None:
                        
            #             # del g.forward_computations[key]['result']
            #             if key in persist_forward_pass_results_list:
            #                 g.forward_computations[key]['result'] = g.forward_computations[key]['result'].detach()
            #             else:
            #                 g.forward_computations[key]['result'] = g.forward_computations[key]['result'].detach()
            #                 g.forward_computations[key]['result'] = None

            if step:
                for key in g.forward_computations.keys():
                    # op_result = g.forward_computations[key]
                    if isinstance(g.forward_computations[key], dict):
                        # op_optimizer = g.forward_computations[key].get('optimizer', None)
                        if g.forward_computations[key].get('optimizer', None) is not None:                            
                            print("\n Step called")
                            g.forward_computations[key]['optimizer'].step()
                            g.forward_computations[key]['optimizer'].zero_grad()

                            # g.forward_computations[key]['optimizer'] = op_optimizer

            results.append(json.dumps({
                'operator': operator,
                'status': 'success',
                'op_id': index['op_id']
            }))
            continue

        if operation_type is not None and operator is not None:
            result_payload = compute_locally(payload=index, subgraph_id=subgraph_id, graph_id=graph_id, retain = index['retain'], to_upload=to_upload, gpu_required = gpu_required)
            if not g.error:
                if result_payload is not None:
                    results.append(result_payload)
            else:
                break
    
        t2 = time.time()
        # print("Time taken for operation: ", t2-t1, ' operator: ', operator)
        
    if not g.error:
        optimized_results_list = []
        for index in data:
            if index['op_id'] in subgraph_outputs:
                to_upload = True
            else:
                to_upload = False
            
            if not index.get('retain', False):
                del_keys.append(index['op_id'])

            if to_upload:
                results_dict = {}

                if index['operator'] == 'forward_pass' and not backward_called:
                    results_dict['result'] = g.forward_computations[index['op_id']]['result']

                    # print("\n Results dict: ", results_dict)
                    # if index['op_id'] not in persist_forward_pass_results_list:
                    #     if results_dict.get('result', None) is not None:
                    #         persisted_result = results_dict['result']
                    # else:
                    if index['op_id'] in persist_forward_pass_results_list:
                        if results_dict.get('result', None) is not None:
                            results_dict['result'] = results_dict['result'].to('cpu')
                            
                    # persisted_result_path = dump_result_data(index['op_id'], persisted_result)

                    file_path = dump_data(index['op_id'], results_dict)
                    
                    # if persisted_result_path is not None:
                    #     persisted_result_path = os.path.basename(persisted_result_path)

                    dumped_result = json.dumps({
                        'file_name': os.path.basename(file_path),
                        # 'persisted_result_file_name': persisted_result_path,
                        "op_id": index['op_id'],
                        "status": "success"
                    })
                    optimized_results_list.append(dumped_result) 
                
                elif index['operator'] == 'forward_pass' and backward_called:
                    model_input_tensors = g.forward_computations[index['op_id']]['inputs']
                    input_indices = g.forward_computations[index['op_id']]['input_indices']
                    input_tensor_id = 0
                    for op_id, index_list in input_indices.items():
                        if len(index_list) > 0:
                            results_dict[op_id] = {}
                            for ind in index_list:
                                if model_input_tensors[input_tensor_id].grad is not None:
                                    results_dict[op_id][ind] = model_input_tensors[input_tensor_id].grad
                                input_tensor_id += 1
                        else:
                            results_dict[op_id] = model_input_tensors[input_tensor_id].grad

                            input_tensor_id += 1


                    # for i in range(len(model_input_tensors)):
                    #     if model_input_tensors[i].grad is not None:
                    #         results_dict[input_index_list[i]] = model_input_tensors[i].grad

                    file_path = dump_data(index['op_id'], results_dict, type='backward')

                    dumped_result = json.dumps({
                        'file_name': os.path.basename(file_path),
                        "op_id": index['op_id'],
                        "status": "success"
                    })
                    optimized_results_list.append(dumped_result)
                    # print("\n Results dict with grad: ", results_dict)
                
                else:
                    # print('\n Forward comp in else: ', g.forward_computations)
                    persisted_result_path = None
                    if isinstance(g.forward_computations[index['op_id']], dict):
                        results_dict['result'] = g.forward_computations[index['op_id']]['result']
                    else:
                        results_dict['result'] = g.forward_computations[index['op_id']]
                    if 'forward_pass' in index["operator"]:
                        if index['op_id'] not in persist_forward_pass_results_list:
                            if results_dict.get('result', None) is not None:
                                del results_dict['result']
                        else:
                            if results_dict.get('result', None) is not None:
                                persisted_result = results_dict['result'].to('cpu')
                                persisted_result_path = dump_result_data(index['op_id'], persisted_result)
                    
                    if isinstance(results_dict, torch.Tensor):
                        results_dict = results_dict.to('cpu')

                    file_path = dump_data(index['op_id'], results_dict)
                    if persisted_result_path is not None:
                        persisted_result_path = os.path.basename(persisted_result_path)

                    dumped_result = json.dumps({
                        'file_name': os.path.basename(file_path),
                        'persisted_result_file_name': persisted_result_path,
                        "op_id": index['op_id'],
                        "status": "success"
                    })
                    optimized_results_list.append(dumped_result)

        optimized_results_list.extend(results) 
        results = optimized_results_list

        for temp_file in os.listdir(FTP_TEMP_FILES_FOLDER):
            if 'temp_' in temp_file:
                file_path = os.path.join(FTP_TEMP_FILES_FOLDER, temp_file)

                try:
                    with zipfile.ZipFile('local_{}_{}.zip'.format(subgraph_id, graph_id), 'a') as zipObj2:
                        zipObj2.write(file_path, os.path.basename(file_path))
                except zipfile.BadZipFile as error:
                    print(error)
                os.remove(file_path)

        zip_file_name = 'local_{}_{}.zip'.format(subgraph_id, graph_id)
        if os.path.exists(zip_file_name):
            g.ftp_client.upload(zip_file_name, zip_file_name)
            os.remove(zip_file_name)

        emit_result_data = {"subgraph_id": d["subgraph_id"], 
                            "graph_id": d["graph_id"], 
                            "token": g.ravenverse_token,
                            "results": results}
        print("\n Results: ", results)
        await g.client.emit("forward_subgraph_completed", json.dumps(emit_result_data), namespace="/client")
        g.logger.debug("Forward subgraph Results Emitted")
        # os.system('clear')
        g.dashboard_data[-1][2] = "Computed"
        print(AsciiTable([['Provider Dashboard']]).table)
        print(AsciiTable(g.dashboard_data).table)

    g.has_subgraph = False
    
    delete_dir = FTP_DOWNLOAD_FILES_FOLDER
    for f in os.listdir(delete_dir):
        os.remove(os.path.join(delete_dir, f))

    g.delete_files_list = []
    # model_key, model_k = None, None
    # for key, val in forward_computations.items():
    #     if isinstance(val, torch.Tensor):
    #         val.detach()
    #     elif isinstance(val, dict):
    #         for k, v in val.items():
    #             if isinstance(v, torch.Tensor):
    #                 v.detach()
    #             elif isinstance(v, torch.nn.Module):
    #                 model_key = key
    #                 model_k = k
        
    # if model_key is not None and model_k is not None:
    #     del forward_computations[model_key][model_k]

    # forward_computations = {}
    # print("\n Data: ", data)
    
    # for key in g.forward_computations.keys():
    #     if key not in retain_keys:
    #         del g.forward_computations[key]

    for key in del_keys:
        if g.forward_computations.get(key, None) is not None:
            del g.forward_computations[key]

    # print("\n G.for comp at end: ", g.forward_computations)
    g.logger.debug('subgraph_forward_process OVER')
    if gpu_required:
        torch.cuda.empty_cache()
    gc.collect()
    return


async def subgraph_backward_process(d):
    # os.system('clear')
    print(AsciiTable([['Provider Dashboard']]).table)
    g.dashboard_data.append([d["subgraph_id"], d["graph_id"], "Backward Computing"])
    print(AsciiTable(g.dashboard_data).table)

    g.logger.debug('subgraph_backward_process received')

    # print("\n Data in backward: ", d)

    g.has_subgraph = True
    subgraph_id = d["subgraph_id"]
    graph_id = d["graph_id"]
    data = d["payloads"]
    subgraph_zip_file_flag = d["subgraph_zip_file_flag"]
    gpu_required = ast.literal_eval(d["gpu_required"])

    if subgraph_zip_file_flag == "True":
        server_file_path = 'zip_{}_{}.zip'.format(subgraph_id, graph_id)

        download_path = os.path.join(FTP_DOWNLOAD_FILES_FOLDER, server_file_path)
        filesize = d['zip_file_size']
        try:
            g.ftp_client.ftp.voidcmd('NOOP')
            g.ftp_client.download(download_path, os.path.basename(server_file_path), filesize)

        except Exception as error:
            os.system('clear')
            g.dashboard_data[-1][2] = "Failed"
            print(AsciiTable([['Provider Dashboard']]).table)
            print(AsciiTable(g.dashboard_data).table)
            g.has_subgraph = False
            
            delete_dir = FTP_DOWNLOAD_FILES_FOLDER
            for f in os.listdir(delete_dir):
                os.remove(os.path.join(delete_dir, f))

            g.delete_files_list = []

            print('Error: ', error)
            emit_error(data[0], error, subgraph_id, graph_id)
            if 'broken pipe' in str(error).lower() or '421' in str(error).lower():
                print(
                    '\n\nYou have encountered an IO based Broken Pipe Error. \nRestart terminal and try connecting again')
                sys.exit()

        if os.path.exists(download_path):
            extract_to_path = FTP_DOWNLOAD_FILES_FOLDER
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_path)

            os.remove(download_path)
            g.ftp_client.delete_file(server_file_path)

    for payload in data:
        compute_backward(payload)

    results = []

    for payload in data:
        results_dict = {}

        model_input_tensors = g.forward_computations[payload['op_id']]['inputs']
        input_indices = g.forward_computations[payload['op_id']]['input_indices']
        input_tensor_id = 0
        for op_id, index_list in input_indices.items():
            if len(index_list) > 0:
                results_dict[op_id] = {}
                for ind in index_list:
                    if model_input_tensors[input_tensor_id].grad is not None:
                        results_dict[op_id][ind] = model_input_tensors[input_tensor_id].grad
                    input_tensor_id += 1
            else:
                results_dict[op_id] = model_input_tensors[input_tensor_id].grad

                input_tensor_id += 1



        # model_input_tensors = g.forward_computations[payload['op_id']]['inputs']
        # input_index_list = g.forward_computations[payload['op_id']]['input_index_list']
        # for i in range(len(model_input_tensors)):
        #     if model_input_tensors[i].grad is not None:
        #         if len(input_index_list) > 0:
        #             results_dict['grad'][input_index_list[i]] = model_input_tensors[i].grad
        #         else:
        #             results_dict['grad'] = model_input_tensors[i].grad

        # results_dict['optimizer'] = g.forward_computations[payload['op_id']]['optimizer']

        file_path = dump_data(payload['op_id'], results_dict, type='backward')
        # model_file_path = dump_torch_model(payload['op_id'], g.forward_computations[payload['op_id']]['instance'])

        dumped_result = json.dumps({
            'file_name': os.path.basename(file_path),
            # 'model_path_name': os.path.basename(model_file_path),
            "op_id": payload['op_id'],
            "status": "success"
        })

        results.append(dumped_result)

    # print('\n Final results list: ', results)

    if not g.error:
        for temp_file in os.listdir(FTP_TEMP_FILES_FOLDER):
            if 'temp_' in temp_file:
                file_path = os.path.join(FTP_TEMP_FILES_FOLDER, temp_file)

                try:
                    with zipfile.ZipFile('local_{}_{}.zip'.format(subgraph_id, graph_id), 'a') as zipObj2:
                        zipObj2.write(file_path, os.path.basename(file_path))
                except zipfile.BadZipFile as error:
                    print(error)
                os.remove(file_path)

        zip_file_name = 'local_{}_{}.zip'.format(subgraph_id, graph_id)
        if os.path.exists(zip_file_name):
            g.ftp_client.upload(zip_file_name, zip_file_name)
            os.remove(zip_file_name)

        emit_result_data = {"subgraph_id": d["subgraph_id"], 
                            "graph_id": d["graph_id"], 
                            "token": g.ravenverse_token,
                            "results": results}
        await g.client.emit("backward_subgraph_completed", json.dumps(emit_result_data), namespace="/client")

        # os.system('clear')
        g.dashboard_data[-1][2] = "Backward Computed"
        print(AsciiTable([['Provider Dashboard']]).table)
        print(AsciiTable(g.dashboard_data).table)

    g.has_subgraph = False
    
    delete_dir = FTP_DOWNLOAD_FILES_FOLDER
    for f in os.listdir(delete_dir):
        os.remove(os.path.join(delete_dir, f))

    g.delete_files_list = []
    # model_key, model_k = None, None
    # for key, val in forward_computations.items():
    #     if isinstance(val, torch.Tensor):
    #         val.detach()
    #     elif isinstance(val, dict):
    #         for k, v in val.items():
    #             if isinstance(v, torch.Tensor):
    #                 v.detach()
    #             elif isinstance(v, torch.nn.Module):
    #                 model_key = key
    #                 model_k = k
        
    # if model_key is not None and model_k is not None:
    #     del forward_computations[model_key][model_k]

    # forward_computations = {}
    # print("\n Data: ", data)
    
    # for key in g.forward_computations.keys():
    #     if key not in retain_keys:
    #         del g.forward_computations[key]

    # print("\n G.for comp at end: ", g.forward_computations)
    g.logger.debug('subgraph_backward_process OVER')
    if gpu_required:
        torch.cuda.empty_cache()
    gc.collect()
    return

@g.client.on('redundant_subgraph', namespace="/client")
async def redundant_subgraph(d):
    subgraph_id = d['subgraph_id']
    graph_id = d['graph_id']
    for i in range(len(g.dashboard_data)):
        if g.dashboard_data[i][0] == subgraph_id and g.dashboard_data[i][1] == graph_id:
            g.dashboard_data[i][2] = "redundant_computation"
    os.system('clear')
    print(AsciiTable([['Provider Dashboard']]).table)
    print(AsciiTable(g.dashboard_data).table)
    return

@g.client.on('share_completed', namespace="/client")
async def share_completed(d):
    print("You have computed your share of subgraphs for this Graph, disconnecting...")
    await exit_handler()
    os._exit(1)


def waitInterval():
    global timeoutId, opTimeout, initialTimeout
    try:
        sock = socket.create_connection(('8.8.8.8',53))
        sock.close()
    except Exception as e:
        print('\n ----------- Device offline -----------')
        os._exit(1)

    if g.client.connected:
        stopTimer(timeoutId)
        timeoutId = setTimeout(waitInterval, opTimeout)

    if not g.is_downloading:
        if not g.is_uploading:
            if g.noop_counter % 17 == 0:
                try:
                    g.ftp_client.ftp.voidcmd('NOOP')

                except Exception as e:
                    print('\n Crashing...')
                    exit_handler()
                    os._exit(1)

    g.noop_counter += 1

async def exit_handler():
    g.logger.debug('Application is Closing!')
    if g.client is not None:
        g.logger.debug("Disconnecting...")
        if g.client.connected:
            await g.client.emit("disconnect", namespace="/client")

    dir = FTP_TEMP_FILES_FOLDER
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))