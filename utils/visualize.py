import imageio
from IPython.display import Image, display
import torch
import matplotlib.pyplot as plt
def gif(data,gif_path,fps):
    frames = []
    ch = data.shape[0]
    for i in range(ch):
        img = data[0]["image"][i, 0]
        img = img - img.min()
        img = img/img.max()
        img_rgb = (img * 255).astype('uint8')
        frames.append(img_rgb)
    imageio.mimsave(gif_path, frames, fps=fps)  # Adjust fps as needed
    display(Image(filename=gif_path))

def img_mask_ori(img,mask):
    # Get the image and mask
    #img, mask = spinal_train_dataset[200]

    # Reorder the dimensions of img from (3, 256, 256) to (256, 256, 3) for displaying
    img = img.permute(1, 2, 0)

    # Convert image tensor to numpy for displaying
    img = img.numpy()

    # Print the shapes for debugging
    height, width = img.shape[:2]
    print(f"Image shape: {img.shape}")  # Should be (256, 256, 3) for 3-channel RGB
    print(f"Mask shape: {mask.shape}")  # Should be (256, 256) for the segmentation mask
    print(f"Unique values in the mask: {torch.unique(mask)}")

    # Visualize the image
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Plot the mask (with integer labels for each class)
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='jet', alpha=0.6)  # Different color for each class
    plt.title("Mask")

    # Overlay mask on the image with transparency
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask on the image
    plt.title("Image with Mask Overlay")

    # Show all the plots
    plt.show()
    

#plot history loss on each iter
def losshistory_plot(train_loss_history, test_loss_history):
    epochs = range(1,len(train_loss_history)+1)
    fig,axe = plt.subplots(1,1,figsize=(10,10))
    axe.plot(epochs, train_loss_history, label='Train Loss')
    axe.plot(epochs, test_loss_history, label = 'Test Loss')
    axe.set_xlabel('Epochs over batch')
    axe.set_ylabel('Loss')
    axe.set_title('Train and val Loss on each batch over epochs')
    axe.legend()
    
    plt.tight_layout()
    #plt.savefig(f'epochs on {model_name}.jpg')
    plt.show()

#plot model prediction on images
def model_plot(model, test_dataset, list_img):
    f, ax = plt.subplots(2, 4, figsize=(13, 8))
    
    for i, img_id in enumerate(list_img):
        img, mask,_ = test_dataset[img_id]
        image = img.permute(1, 2, 0)
        height, weight = image.shape[:2]
        
        # Original image with label scatter plot
        ax[0, i].imshow(image, cmap='gray')
        ax[0, i].imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask on the image

        # Model prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        img = img.unsqueeze(0).to(device)
        output = model(img).cpu().detach() 
        _, outputs_mask = torch.max(output, 1)
        pred_mask = outputs_mask.squeeze(0).numpy() 
        print(pred_mask.shape)
        ax[1, i].imshow(image, cmap='gray')
        ax[1, i].imshow(pred_mask, cmap='jet', alpha=0.5)
    
    plt.tight_layout()
    plt.show()