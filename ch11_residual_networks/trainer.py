import torch
import tqdm
import time

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion,
                 device=None, scheduler=None, gradient_fn=None,
                 enable_gradient_tracking=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epoch_times = []

        self.gradient_fn = gradient_fn
        self.enable_gradient_tracking = enable_gradient_tracking and gradient_fn is not None
        self.gradient_history = None
        self.module_layer_length = 0

        if self.enable_gradient_tracking:
            self._setup_gradient_tracking()

    def _setup_gradient_tracking(self):
        try:
            self.model.train() # enable train so we store the grads and detect the number of layers

            # testing it with one sample input so we can test the gradient tracking and get the number of layers being tracked
            sample_batch = next(iter(self.train_loader))
            sample_input = sample_batch[0][:1].to(self.device)
            dummy_output = self.model(sample_input)
            dummy_loss = dummy_output.mean()
            dummy_loss.backward()

            if self.gradient_fn is None:
                raise NotImplementedError("gradient_fn not implemented.")

            gradients_per_layer = self.gradient_fn(self.model)
            self.optimizer.zero_grad()

            self.module_layer_length = len(gradients_per_layer)
        except Exception as e:
            print(f"⚠️  Gradient tracking setup failed: {e}")
            print("🔄 Disabling gradient tracking...")
            self.enable_gradient_tracking = False
            self.module_layer_length = 0  # Set default


    def train_epoch(self):
        batch_gradients = []
        if self.enable_gradient_tracking:
            batch_gradients = torch.zeros((len(self.train_loader), self.module_layer_length))

        self.model.train()

        total_correct = 0
        total_items = 0
        total_loss = 0
        start_time = time.time()
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.train_loader), "Training"):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            if self.enable_gradient_tracking:
                grad_mags = self.gradient_fn(self.model)
                batch_gradients[batch_idx] = grad_mags


            self.optimizer.step()

            _, predicted = output.max(1)
            correct = (predicted == target).sum().item()
            total_loss += loss.item()
            total_correct += correct
            total_items += target.size(0)

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = total_correct / total_items

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "time": epoch_time,
            "gradients": batch_gradients.mean(dim=0) if self.enable_gradient_tracking else None,
        }

    def val_epoch(self):
        self.model.eval()
        total_correct = 0
        total_items = 0
        total_loss = 0
        start_time = time.time()


        with torch.no_grad():
            for data, target in tqdm.tqdm(self.val_loader, "Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                _, predicted = output.max(1)

                correct = (target == predicted).sum().item()
                total_correct += correct
                total_loss += loss.item()
                total_items += target.size(0)

        val_acc = total_correct / total_items
        val_loss = total_loss / len(self.val_loader)
        val_time = time.time() - start_time

        return {
            "loss": val_loss,
            "acc": val_acc,
            "time": val_time,
        }

    def fit(self, epochs):
        if self.enable_gradient_tracking:
            self.gradient_history = torch.zeros((epochs, self.module_layer_length))
        else:
            self.gradient_history = None

        print("\n" + "=" * 70)
        print(f"{'🚀 TRAINING STARTED':^70}")
        print("=" * 70 + "\n")

        for epoch in range(epochs):
            # Epoch header
            print(f"\n{'─' * 70}")
            print(f"📍 Epoch {epoch + 1:3d}/{epochs}")
            print(f"{'─' * 70}")

            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()

            self.train_losses.append(train_metrics["loss"])
            self.train_accs.append(train_metrics["acc"])
            self.val_losses.append(val_metrics["loss"])
            self.val_accs.append(val_metrics["acc"])
            self.epoch_times.append(val_metrics["time"] + train_metrics["time"])

            if self.enable_gradient_tracking and train_metrics["gradients"] is not None:
                self.gradient_history[epoch] = train_metrics["gradients"]

            # Calculate metrics
            total_time = train_metrics["time"] + val_metrics["time"]

            # Pretty print results
            print(f"\n📊 Results:")
            print(f"   Train → Loss: {train_metrics['loss']:7.4f}  |  Acc: {train_metrics['acc'] * 100:6.2f}%")
            print(f"   Val   → Loss: {val_metrics['loss']:7.4f}  |  Acc: {val_metrics['acc'] * 100:6.2f}%")
            print(f"   ⏱️  Time: {total_time:.2f}s")

            # Show improvement indicators
            if epoch > 0:
                acc_change = (self.val_accs[-1] - self.val_accs[-2]) * 100
                loss_change = self.val_losses[-1] - self.val_losses[-2]

                acc_arrow = "↗️" if acc_change > 0 else "↘️" if acc_change < 0 else "→"
                loss_arrow = "↗️" if loss_change > 0 else "↘️" if loss_change < 0 else "→"

                print(f"   Change: Acc {acc_arrow} {acc_change:+.2f}%  |  Loss {loss_arrow} {loss_change:+.4f}")

            # Show gradient info if tracking
            if self.enable_gradient_tracking and train_metrics["gradients"] is not None:
                grad_mean = train_metrics["gradients"].mean().item()
                grad_std = train_metrics["gradients"].std().item()
                grad_min = train_metrics["gradients"].min().item()
                grad_max = train_metrics["gradients"].max().item()
                print(f"   📉 Gradients: μ={grad_mean:.2e}, σ={grad_std:.2e}, range=[{grad_min:.2e}, {grad_max:.2e}]")

            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   📚 Learning Rate: {current_lr:.6f}")
                self.scheduler.step()

        # Training complete summary
        print(f"\n{'=' * 70}")
        print(f"{'✅ TRAINING COMPLETE':^70}")
        print(f"{'=' * 70}")
        print(f"\n📈 Final Results:")
        print(
            f"   Best Val Acc:  {max(self.val_accs) * 100:.2f}% (epoch {self.val_accs.index(max(self.val_accs)) + 1})")
        print(f"   Final Val Acc: {self.val_accs[-1] * 100:.2f}%")
        print(f"   Final Val Loss: {self.val_losses[-1]:.4f}")
        print(f"   Total Time: {sum(self.epoch_times) / 60:.1f} minutes")
        print(f"{'=' * 70}\n")


