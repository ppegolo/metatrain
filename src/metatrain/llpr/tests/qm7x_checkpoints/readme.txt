Backbone checkpoints for test_qm7x_regression.py. They live in this separate
folder (not in checkpoints/) because the checkpoint-compatibility tests load
everything in checkpoints/*.ckpt.gz as LLPR-architecture checkpoints.

Both models were trained on the first 90 structures of
tests/resources/qm7x_spherical_100.zip (the last 10 are the held-out
calibration set, see _get_qm7x_datasets() in test_qm7x_regression.py) on all
four targets: energy, non_conservative_force, mtt::dipole and
mtt::polarizability.

To regenerate after a PET or SPACE checkpoint version bump, train with the
architecture's Python API (torch.manual_seed(42), CPU, first supported dtype,
default hypers except the ones below), save with trainer.save_checkpoint and
gzip the result here. Then recompute the EXPECTED_MEAN_UNCERTAINTIES table in
test_qm7x_regression.py from the new checkpoints.

Common training hypers:
    batch_size: 10
    num_epochs: 1000
    loss: metatrain defaults, one entry per target

pet-qm7x.ckpt.gz (architecture "pet"), learning_rate: 3e-4, model hypers:
    d_pet: 16, d_head: 16, d_node: 16, d_feedforward: 32,
    num_heads: 2, num_attention_layers: 2, num_gnn_layers: 2

space-qm7x.ckpt.gz (architecture "experimental.space"), model hypers:
    num_element_channels: 4, num_gnn_layers: 1, num_tensor_products: 2,
    mlp_head_expansion_ratio: 1,
    radial_basis: {max_eigenvalue: 25.0, mlp_expansion_ratio: 1, mlp_depth: 2}
