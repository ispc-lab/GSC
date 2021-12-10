name = "STAttnsGraphDashcam"
version = "2.0.0"
model = dict(
    type='STAttnsGraphSimAllBatch',
    spatial_graph=dict(
        type='SpatialGraphBatch',
        node_feature=2048,
        hidden_feature=1024,
        out_feature=512,
    ),
    temporal_graph=dict(
        type='TemporalGraphBatch',
        past_num=5,
        input_feature=512,
        out_feature=512,
    ),
    gate=dict(
        type='MaskGate',
        past_num = 10,
    ),
    temporal_attn=dict(
        type='STAttn'
    ),
    near_future=dict(
        type="NearFuture"
    ),
    accident_block=dict(
        type='AccidentLSTM',
        temporal_feature=256,
        hidden_feature=64,
        num_layers=2
    ),
    loss=dict(
        type="LogLoss"
    ))
dataset_type = "DashCam"
data_root = "/media/group1/data/tianhang/MASKER_MD" 
data = dict(
    batch_size=32,
    num_workers=1,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        #pipelines=pipelines,
        video_list_file=f"{data_root}/train_video_list.txt"),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        #pipelines=pipelines,
        video_list_file=f"{data_root}/valid_video_list.txt"))
# training and testing settings
optimizer_cfg = dict(
    type='Adam',
    lr=0.01,
    betas=(0.9, 0.999),
)
lr_cfg = dict(
    type="StepLR",
    step_size=50,
    gamma=.7)
warm_up_cfg = dict(
    type="Exponential",
    step_size=5000)
random_seed = 1234
# GPU 
num_gpus = [2]
max_epochs = 200
checkpoint_path = "work_dirs/checkpoints"
log_path = "work_dirs/logs"
result_path = "work_dirs/results"
load_from_checkpoint = None
resume_from_checkpoint = None
test_checkpoint = None
batch_accumulate_size = 1
simple_profiler = True