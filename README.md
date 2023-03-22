<div align="center">
  <img src="https://user-images.githubusercontent.com/36446402/217179360-b39cf6b8-456c-4d40-bedc-afe8eee62611.svg" width="200" height="100">
<h1> Ravpy - The Provider's Library </h1>
</div>

Introducing Raven Protocol's Python SDK for Providers that allows compute nodes across the world to participate in the Ravenverse. 

Providers are those who are willing to contribute their local system's idle compute power to execute the requester's computational requirements in return for rewards (Raven Tokens).

Ravpy is a python SDK that allows providers to intuitively participate in any ongoing graph computations in the Ravenverse. Ravpy nodes are authenticated based on their Ravenverse tokens which must be first generated by visiting the [Ravenverse website](https://www.ravenverse.ai/).

Ravpy connects to Ravsock (backbone server) from which it receives subgraphs (small groupings of Ravop operations) based on a benchmarking procedure. During this procedure, Ravpy client systems are rigorously tested on a variety of operations and the results are utilized by Ravsock for efficiently assigning complex subgraphs to the provider.  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Installation

### With Pip

```bash
pip install ravpy
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)
## Usage

This section details how to join Ravenverse as a Provider. The relevant scripts are available in the [Ravenverse Repository](https://github.com/ravenprotocol/ravenverse). It is recommended to clone the Ravenverse repo and navigate to the ```Provider``` folder, where scripts for following steps are available.

### Setting Environment Variables
Create a .env file and add the following environment variables:

```bash
RAVENVERSE_URL=http://server.ravenverse.ai
RAVENVERSE_FTP_HOST=server.ravenverse.ai
RAVENVERSE_FTP_URL=server.ravenverse.ai
```

Load environment variables at the beginning of your code using:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Authentication

The Provider must connect to the Ravenverse using a unique token that they can generate by logging in on the Ravenverse Website using their MetaMask wallet credentials.   

```python
from ravpy.initialize import initialize

initialize(ravenverse_token = '<token>')
```

### Participate in Distributed Computing
A Provider can choose which graph to participate in. In order to list which graphs are available, the following command can be used:

```python
from ravpy.utils import list_graphs

list_graphs(approach="distributed")
```

<img width="1423" alt="Screenshot 2023-03-22 at 11 22 44 AM" src="https://user-images.githubusercontent.com/36446402/226816986-d28fbacb-aed6-4113-b808-43d0153f9521.png">


This command displays a list of executed graphs along with the minimum requirements to participate in their computation, including system requirements and minimum stake required. A provider can participate in a graph only if their system meets these requirements and they have sufficient tokens in their account to meet the minimum stake amount.<br><br>
The following command can be run to participate in a graph (the Provider must note the id of the graph they wish to participate in):

```python
from ravpy.distributed.participate import participate

participate(graph_id=1)
```

Upon participation, the Provider will be assigned a number of subgraphs to execute. Once they have been executed, the results will be sent to the Ravsock server. The full staked amount will be returned along with their earnings on successful computation of all assigned graphs. <br>However if the provider disconnects before the computation of their share of subgraphs, a portion of the staked amount will be deducted.

![ezgif com-optimize](https://user-images.githubusercontent.com/36446402/226816757-6381583b-9dc1-4af5-b3e5-9d3886ab8ff3.gif)


### Participate in Federated Analytics

```python
from ravpy.federated.participate import participate

participate(graph_id=3, file_path="<file_path>")
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

<!-- ## How to Contribute -->

## License

<a href="https://github.com/ravenprotocol/ravpy/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/ravpy"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
