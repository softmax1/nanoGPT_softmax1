FROM python:3.9
RUN pip install torch numpy transformers datasets tiktoken wandb tqdm boto3[crt]
WORKDIR /
RUN git clone https://github.com/mcapodici/nanoGPT nanoGPT && \
    cd nanoGPT && \
    git checkout softmax1 
        

