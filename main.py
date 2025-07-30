import jax                                                        # JAX 核心庫，用來控制隨機性和加速數值運算
import jax.numpy as jnp                                           # JAX 的 NumPy 版本，用來處理張量運算
import matplotlib.pyplot as plt                                   # 畫圖用，視覺化預測結果
import numpy as np                                                # 輔助進行數值操作
from model import MlpMixer                                        # 匯入你自己定義的子模組：主模型架構
from dataset import load_dataset                                  # 載入訓練與測試資料
from train import create_train_state, train_step                  # 建立初始訓練狀態（包含 optimizer）、單一訓練步驟邏輯
from eval import evaluate                                         # 執行推論並計算準確率
from utils import plot_metrics                                    # 顯示訓練過程的準確率與損失趨勢

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',       # 建立 CIFAR-10 的 10 類分類標籤，用來顯示預測結果時的名稱
           'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess(image):                                            # 將影像轉為 float32 並正規化到 [0, 1] 範圍，方便模型訓練
    return jnp.array(image, dtype=jnp.float32) / 255.0

def visualize_prediction(image, pred_class, true_class):          # 顯示單張影像並標示「預測類別 vs. 正確類別」，幫助分析模型表現
    plt.imshow(image.astype(np.float32))
    plt.title(f"Prediction: {classes[pred_class]}\nGround Truth: {classes[true_class]}")
    plt.axis('off')
    plt.show()

def run_multiple_test_samples(model, params, test_data, num_samples=10):           # 從測試資料中隨機挑選 num_samples 張圖進行推論
    print(f"\n📷 Running inference on {num_samples} random test images...")
    correct = 0
    error_stats = {}

    for _ in range(num_samples):
        imgs, labels = test_data[np.random.randint(len(test_data))]
        image = imgs[0]
        label = int(labels[0])

        image_norm = image[None, ...]
        logits = model.apply(params, image_norm)
        pred_class = int(jnp.argmax(logits, axis=-1)[0])

        visualize_prediction(image, pred_class, label)
        print(f"🔹 Predicted: {classes[pred_class]}")
        print(f"🔸 Ground Truth: {classes[label]}\n")

        # 記錄正確率
        if pred_class == label:
            correct += 1
        else:
            true_label = classes[label]
            pred_label = classes[pred_class]
            error_stats.setdefault((true_label, pred_label), 0)
            error_stats[(true_label, pred_label)] += 1

    acc = correct / num_samples
    print(f"✅ Correct Predictions: {correct}/{num_samples}")
    print(f"📈 Accuracy: {acc:.2f}")

    if error_stats:
        print("\n❌ Misclassifications:")
        for (true_cls, pred_cls), count in error_stats.items():
            print(f"- {true_cls} → {pred_cls}: {count} time(s)")

# ...existing code...

def main():
    rng = jax.random.PRNGKey(0)

    model = MlpMixer(
        num_classes=10,
        num_blocks=4,
        patch_size=4,
        hidden_dim=64,
        tokens_mlp_dim=128,
        channels_mlp_dim=256,
    )

    batch_size = 128
    num_epochs = 20

    train_data = load_dataset(batch_size=batch_size, train=True)
    test_data = load_dataset(batch_size=batch_size, train=False)

    steps_per_epoch = len(train_data)

    state = create_train_state(
        rng, model,
        learning_rate=0.001,           # 可微調
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch
    )
    accs, losses = [], []

    for epoch in range(num_epochs):
        for batch in train_data:
            state, metrics = train_step(state, batch)

        accs.append(metrics['accuracy'])
        losses.append(metrics['loss'])
        print(f"Epoch {epoch+1} — Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")

    test_acc = evaluate(model, state.params, test_data)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    plot_metrics(accs, "Accuracy")
    plot_metrics(losses, "Loss")

    print("\n🔍 Multiple image inference from test set:")
    run_multiple_test_samples(model, state.params, test_data, num_samples=10)

if __name__ == "__main__":
    main()