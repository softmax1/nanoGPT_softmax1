FROM python:3.9

#    .apt_install("git") \
RUN pip install torch numpy transformers datasets tiktoken wandb tqdm
WORKDIR /
RUN git clone https://github.com/karpathy/nanoGPT nanoGPT && \
    cd nanoGPT && \
    git checkout eba36e84649f3c6d840a93092cb779a260544d08 
        

