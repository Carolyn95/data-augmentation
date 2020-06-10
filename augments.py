import ipywidgets as widgets
import requests, os
from IPython.display import display
from ipywidgets import interact
import tqdm
from rasa.nlu.training_data import TrainingData, Message
from model import ParaphraseModel

model_path = 'package_models/lm_finetune_8/checkpoint-56000/'

complete_td = TrainingData()
model = ParaphraseModel(model_path)

input_phrase = input(
    "Enter a message for which you would like to generate paraphrases: ")

number_samples = int(input("Number of paraphrases to generate: "))
stop_words = input(
    "Stop words to be constrained with(multiple semi-colon separated): ")

paraphrases = model.get_paraphrases(input_phrase, number_samples, stop_words)

print(
    "Steps:\n1. Read all proposed paraphrases below.\n2. Select valid paraphrases that you would like\
 to add to your NLU training data. Use Ctrl/Cmd + Click to select multiple.\n\
3. Enter the name of the intent under which these messages should be categorized\n\
4. Click 'Add to training data'\n\
5. Copy the training data displayed in Rasa Markdown format to your existing training data file.\n\
6. You can go back to 3 cells above this to enter new messages for which you want to generate paraphrases."
)

paraphrase_widget = widgets.SelectMultiple(options=paraphrases,
                                           value=[],
                                           rows=number_samples,
                                           description='Paraphrases',
                                           disabled=False,
                                           layout=widgets.Layout(width='100%'))
display(paraphrase_widget)

intent = widgets.Text(description="Intent")
display(intent)

button = widgets.Button(description="Add to Training Data")
output = widgets.Output()

display(button, output)


def on_button_clicked(b):

  global complete_td

  with output:
    intent_value = intent.value
    selected_paraphrases = paraphrase_widget.value

    if not len(selected_paraphrases):
      print("Error: You haven't selected any paraphrases")
      return
    if not intent_value:
      print(
          "Error: Please enter the intent name under which these messages should be categorized."
      )
      return

    all_messages = [Message.build(text=input_phrase, intent=intent_value)]
    for paraphrase in selected_paraphrases:
      all_messages.append(Message.build(text=paraphrase, intent=intent_value))

    complete_td = complete_td.merge(
        TrainingData(training_examples=all_messages))

    print(complete_td.nlu_as_markdown())


button.on_click(on_button_clicked)
