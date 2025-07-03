import os
import pickle
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey there! 👋\n\n"
        "I'm Iris Classifier Bot.\n"
        "Just send me 4 numbers, separated by spaces:\n"
        "_Example:_ `5.1 3.5 1.4 0.2`",
        parse_mode="Markdown"
    )

async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    parts = text.split()

    if len(parts) != 4:
        await update.message.reply_text(
            "Please send exactly *4 numbers*, separated by spaces.\n\n"
            "_Example:_ `6.1 2.8 4.7 1.2`",
            parse_mode="Markdown"
        )
        return

    try:
        values = [float(x) for x in parts]
    except ValueError:
        await update.message.reply_text("Oops! Please make sure all values are numbers.")
        return

    X = [values]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][pred]

    flower = CLASS_NAMES[pred]
    confidence = round(proba * 100, 2)

    text = (
        f"🌸 Predicted Flower: *{flower}*\n"
        f"🔎 Confidence: *{confidence}%*"
    )

    keyboard = [
        [InlineKeyboardButton("🔁 Try Again", callback_data="restart")],
        [InlineKeyboardButton("📊 Example Inputs", callback_data="example")],
        [InlineKeyboardButton("ℹ️ About", callback_data="about")],
    ]

    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "restart":
        await query.message.reply_text(
            "Alright, send me 4 new numbers.\n\n"
            "_Example:_ `6.3 3.3 6.0 2.5`",
            parse_mode="Markdown"
        )
    elif query.data == "example":
        await query.message.reply_text(
            "📊 Here are some example inputs you can try:\n\n"
            "`5.1 3.5 1.4 0.2`\n"
            "`6.7 3.0 5.2 2.3`\n"
            "`4.9 2.5 4.5 1.7`",
            parse_mode="Markdown"
        )
    elif query.data == "about":
        await query.message.reply_text(
            "ℹ️ I'm a simple bot that predicts the type of Iris flower 🌸\n"
            "based on four measurements you give me.\n\n"
            "Made with ❤️ using Python and scikit-learn.",
            parse_mode="Markdown"
        )

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_input))

    print("✅ Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
