# #7565910815:AAHSAT6AtHwhBdcqtLdvrIAjhI_6ZRrKRUc

import os
import cv2
import pytesseract
from gtts import gTTS
from ultralytics import YOLO
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import logging

# Paths
model_path = "best_30_epochs.pt"  # Update this to your local model path
output_dir = "cropped_boxes"

# Initialize YOLO model
model = YOLO(model_path)

# Set up logging for the bot
logging.basicConfig(level=logging.INFO)

# Define language choices for the bot
def set_language(update: Update, context: CallbackContext) -> None:
    reply_keyboard = [['English', 'Hindi']]
    update.message.reply_text(
        "Please choose the language of the newspaper:",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

# Handle the user’s language choice
def handle_language_choice(update: Update, context: CallbackContext) -> None:
    choice = update.message.text.lower()
    
    if choice in ['english', 'hindi']:
        context.user_data['language'] = choice
        update.message.reply_text(f"Great! You've selected {choice.title()}. Now, please upload the newspaper image as a file for better results.")
    else:
        update.message.reply_text("Invalid choice. Please select either 'English' or 'Hindi'.")

# Define the start function for the bot
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "Hi! Send me an image of a newspaper, and I'll process it. "
        "For better reading results, please upload the image as a file instead of a photo. "
        "To start, please set the language of the newspaper by using the /setlanguage command."
    )

# Define the image handler for processing images sent as a photo
def handle_image(update: Update, context: CallbackContext) -> None:
    if 'language' not in context.user_data:
        update.message.reply_text("Please choose the newspaper language first by using /setlanguage.")
        return
    
    # Get the image from the user (for photo type)
    image = update.message.photo[-1].get_file()
    input_image_path = os.path.join(output_dir, "input_image.jpg")
    image.download(input_image_path)
    
    process_image(update, context, input_image_path)

# Define the document handler for processing images sent as a file
def handle_document(update: Update, context: CallbackContext) -> None:
    if 'language' not in context.user_data:
        update.message.reply_text("Please choose the newspaper language first by using /setlanguage.")
        return
    
    # Get the file if it's an image
    file = update.message.document
    if file.mime_type.startswith("image/"):
        input_image_path = os.path.join(output_dir, file.file_name)
        file = file.get_file()
        file.download(input_image_path)
        
        process_image(update, context, input_image_path)
    else:
        update.message.reply_text("Please upload a valid image file.")

# Function to process images based on language choice
def process_image(update: Update, context: CallbackContext, input_image_path: str) -> None:
    try:
        # Perform inference using YOLO
        results = model(input_image_path)
        original_image = cv2.imread(input_image_path)

        extracted_texts = []
        selected_language = context.user_data.get('language', 'hindi')  # Default to Hindi if not set

        for i, box in enumerate(results[0].boxes.xyxy):
            try:
                x1, y1, x2, y2 = map(int, box)
                confidence = results[0].boxes.conf[i]
                class_id = int(results[0].boxes.cls[i])  # Get class id of the detected object
                class_name = model.names[class_id]  # Get the class name

                if confidence > 0.1 and class_name == 'news':  # Only process if it's a 'news' box
                    # Crop the image
                    cropped_img = original_image[y1:y2, x1:x2]
                    cropped_img_path = os.path.join(output_dir, f"cropped_box_{i}.jpg")
                    cv2.imwrite(cropped_img_path, cropped_img)

                    # Send the cropped image and class information back to the user
                    context.bot.send_photo(chat_id=update.message.chat_id, photo=open(cropped_img_path, 'rb'))
                    update.message.reply_text(f"Processing detected news article")

                    # OCR and text extraction based on the chosen language
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    if selected_language == 'english':
                        text = pytesseract.image_to_string(cropped_img_rgb, lang='eng')
                    else:  # Hindi
                        text = pytesseract.image_to_string(cropped_img_rgb, lang='hin')

                    cleaned_text = "\n".join(line.strip().replace(":", "") for line in text.splitlines() if line.strip())
                    extracted_texts.append(cleaned_text)

                    # Convert text to audio and send back based on the chosen language
                    if cleaned_text.strip():
                        audio_file = os.path.join(output_dir, f"output_box_{i}.mp3")
                        if selected_language == 'english':
                            tts = gTTS(text=cleaned_text, lang='en')
                        else:  # Hindi
                            tts = gTTS(text=cleaned_text, lang='hi')
                        
                        tts.save(audio_file)
                        context.bot.send_audio(chat_id=update.message.chat_id, audio=open(audio_file, 'rb'))

            except Exception as e:
                print(f"Error processing box {i}: {e}")

    except Exception as e:
        print(f"Error processing image: {e}")
        update.message.reply_text("There was an error processing the image.")

# Main function to set up the bot
def main() -> None:
    # Your Telegram Bot API Token
    TELEGRAM_BOT_TOKEN = '7565910815:AAHSAT6AtHwhBdcqtLdvrIAjhI_6ZRrKRUc'  # Replace with your actual token

    # Set up the bot
    updater = Updater(TELEGRAM_BOT_TOKEN)

    # Register handlers
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("setlanguage", set_language))
    dp.add_handler(MessageHandler(Filters.text & Filters.regex('^(English|Hindi)$'), handle_language_choice))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    dp.add_handler(MessageHandler(Filters.document.category("image"), handle_document))

    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    main()
