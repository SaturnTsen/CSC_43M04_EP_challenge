from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F


def show_images(data_loader, num_images=5, name="sanity_check"):
    """Display a few images from the data loader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data loader to get the images from.
    num_images : int
        The number of images to display.
    name : str
        The name of the file to save the plot to.
    """
    # - get the first batch of images
    iteration = iter(data_loader)
    # inputs = next(iteration)
    # - create the plot
    rows = int(num_images**0.5)
    cols = (num_images // rows) + (num_images % rows > 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, squeeze=True, figsize=(15, 15))
    axs = axs.flatten()

    # - show the images
    for i in range(num_images):
        inputs = next(iteration)["image"]
        i_un = inputs[i].detach().cpu()
        i_img = i_un * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        i_img = F.to_pil_image(i_img)
        axs[i].imshow(i_img)
        axs[i].axis("off")

    # - remove any unused subplots
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])
    plt.close(fig)
    return fig
from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F


def show_images(data_loader, num_images=5, name="sanity_check"):
    """Display a few images from the data loader.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data loader to get the images from.
    num_images : int
        The number of images to display.
    name : str
        The name of the file to save the plot to.
    """
    # 检查数据加载器是否为空
    if len(data_loader) == 0:
        print(f"Warning: {name} data loader is empty, skipping sanity check")
        # 创建一个空的图像作为占位符
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.text(0.5, 0.5, f"No data in {name}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        plt.close(fig)
        return fig
    
    # - get the first batch of images
    iteration = iter(data_loader)
    
    # - create the plot
    rows = int(num_images**0.5)
    cols = (num_images // rows) + (num_images % rows > 0)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, squeeze=True, figsize=(15, 15))
    if num_images == 1:
        axs = [axs]  # 确保axs总是列表
    else:
        axs = axs.flatten()

    # - show the images
    images_shown = 0
    current_batch = None
    
    try:
        for i in range(num_images):
            # 尝试获取新批次或使用当前批次
            if current_batch is None or i >= len(current_batch["image"]):
                try:
                    current_batch = next(iteration)
                    batch_idx = 0
                except StopIteration:
                    # 如果没有更多数据，显示占位符
                    axs[i].text(0.5, 0.5, "No more data", ha='center', va='center', transform=axs[i].transAxes)
                    axs[i].axis("off")
                    continue
            else:
                batch_idx = i % len(current_batch["image"])
            
            # 显示图像
            try:
                i_un = current_batch["image"][batch_idx].detach().cpu()
                i_img = i_un * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406]
                ).view(3, 1, 1)
                i_img = F.to_pil_image(i_img)
                axs[i].imshow(i_img)
                axs[i].axis("off")
                images_shown += 1
            except (IndexError, RuntimeError) as e:
                axs[i].text(0.5, 0.5, f"Error: {str(e)[:20]}...", ha='center', va='center', transform=axs[i].transAxes)
                axs[i].axis("off")
                
    except Exception as e:
        print(f"Warning: Error in {name} sanity check: {e}")

    # - remove any unused subplots
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.close(fig)
    return fig