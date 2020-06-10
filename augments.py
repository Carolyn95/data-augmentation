import tqdm
from rasa.nlu.training_data import TrainingData, Message
from model import ParaphraseModel
import pdb
import numpy as np


def getParaphrases(num_samples, input_sentences, stopwords=' '):
  model_path = 'package_models/lm_finetune_8/checkpoint-56000/'
  complete_td = TrainingData()
  model = ParaphraseModel(model_path)
  stopwords = stopwords
  paraphrases = []
  for sent in input_sentences:
    paraphrases = paraphrases + model.get_paraphrases(sent, num_samples,
                                                      stopwords)
  paraphrases_ = np.array(paraphrases)
  paraphrases_labels = np.array([np.zeros(3) for n in range(len(paraphrases_))])
  assert len(paraphrases_) == len(paraphrases_labels) == 600
  np.save('processed_data/paraphrases.npy', paraphrases_)
  np.save('processed_data/paraphrases_labels.npy', paraphrases_labels)


if __name__ == '__main__':
  input_sentences = [
      "Apologies, I am away and will be back on 30 December 2019",
      "I am out of office until 16 Dec. Will back to office on 17 Dec. Please contact \
__EMAILADDRESS__ for any support issue",
      "This is the mail delivery agent at Symantec Email Security. cloud. I was unable \
to deliver your message to the following addresses:",
      "Thank you for contacting us. We are currently looking into your feedback/enquiry \
and will respond to you within 5 working days. Should you require urgent \
assistance, please contact our hotline at 6213 8008 from Mondays to Fridays, 9 \
am to 5. 30 pm, and Saturdays, 9 am to 12. 30 pm ",
      "I am out of office. For urgent matters, please contact my Admin colleagues for \
assistance",
      "There has been an update via email to Incident Number: INC000004971562. Please \
check via SMoD portal to know the updates. Thanks ",
      "Please leave me a message and I’ll get back to you soon. For any urgent, please \
contact: ", "Delivery has failed to these recipients or groups: ",
      "The following message to __EMAILADDRESS__ was undeliverable. The reason for the \
problem:. 5. 4. 7 - Delivery expired (message too old) 'timeout' ",
      "The following message to __EMAILADDRESS__ was undeliverable. The reason for the \
problem:. 5. 1. 0 - Unknown address error 550-'Invalid recipient \
__EMAILADDRESS__ ",
      "Please note that case INC000004331925 is at 70% milestone for the resolution \
time. ",
      "Please note that case INC000005170800 is at 70% milestone for the resolution \
time. IT Project/Service    NTUC. Number    INC000005170800. Summary Unable to connect sntuc wifi. Location  SPWU. Opened    2020-01-22 16:36:03 SGT. \
State  Assigned. Assignment group      NTUC-I-GSD-ServiceDesk. Assigned To CLEF NG JUN FENG. Description. **** DISCLAIMER ***** ",
      "Lulu Goh - OTCi would like to recall the message, \"NTUC Incident INC000004437022 \
- Update member list for DL : Everyone – UnionL3\" & \"Everyone – UnionGS\" receipt \
confirmation. \"",
      "An email message has been blocked and quarantined by the FireEye Email Malware \
Protection System. The message details are summarized below: ",
      "I am out of office. For urgent matters, you may wish to email __EMAILADDRESS__  \
or contact her at __PHONENUMBER__ ",
      "I am on leave till 26 Dec. Please expect delay in my response. Thanks ",
      "The following message to __EMAILADDRESS__ was undeliverable. The reason for the \
problem:. 5. 1. 2 - Bad destination host 'DNS Hard Error looking up nutc. org. \
sg (MX):  NXDomain' ",
      "I am out of office and will be back on 20 Jan. For system related matters, \
please contact UXI Team at __EMAILADDRESS__ For technical issues on UCEM, please \
email to NTUC Service Desk __EMAILADDRESS__ and UCEM Support Thanks Connie ",
      "I am out of office till 13 Dec and will revert to you when I am back. Have a \
nice day ",
      "Please note that your NCS Mail/LAN password will expire in 14 days. You are \
advised to change your password immediately to prevent logon failure. For \
computer that joins NCS domain, please press CTRL-ALT-DEL from your computer and \
choose Change Password. For other computers, please change via https://webmail. \
ncs. com. sg/owa/auth/expiredpassword. aspx. url=/owa/auth. owa. Password must \
contain character combinations from at least three of the following categories \
(e. g. 2bn_oW). >  One alphabetic upper case character  (A-Z). >   One \
alphabetic lower case character (a-z) ",
      "I am currently on leave will return back on 29 Jan 2020 and I have limited email \
access will reply you as soon as possible ",
      "Thank you for keeping me informed. We are migrating our emails over the weekend, \
and may not respond to your email promptly. For urgent matters, please contact \
me at __PHONENUMBER__. We apologise for any inconvenience caused. Thank you ",
      "Lim Hui Lian - MSD would like to recall the message, Data Patch Request - Void \
backdated payment record and patch join union date and backdated fields due to wrong join union date backdated",
      "I'm currently out of office. Will reply when I'm back to office on 11 Dec 2019 ",
      "My Automatic Replies (Out of Office) does not have a OK box at the bottom of the \
page to click. Is there a way to set it up ",
      "This is the mail delivery agent at Symantec Email Security. cloud. I was unable \
to deliver your message to the following addresses: ",
      "I'm on holiday vacation. For NTUC support, please contact Maria __EMAILADDRESS__",
      "I am out of office. Please expect slower response and I will get back to you as \
soon as I can. For IT support related issues, kindly email __EMAILADDRESS__ or \
call __PHONENUMBER__. Thank you ",
      "Lum Mei Fen - e2i would like to recall the message, \"Updating of email address \
in global directory\" ",
      "I am currently away and have limited access to my email. For urgent matters, \
please feel free to watsapp me. Thank you ",
      "I am on medical leave. Please expect delay in response. Thank you ",
      "Thank you for your email. I will be on leave till 17 Dec 2019. Should you need \
urgent attention to your email, please call __PHONENUMBER__ for assistance. Otherwise, \
I will get back to you on Wed, 18 Dec 2019. Thank you ",
      "Thank you for your email. I am on leave. I will be back in office Tomorrow. \
Please expect some delay in my response as I have limited access to my email ",
      "You are receiving this email because your password for Window login to the account \
\"NCS\ntuc_servicedesk\" is expiring. If your password is not changed before the \
expiry date, you will not be able to login to your laptop or desktop.",
      "Out of office and back in office on Tues 24th Dec. Email responses will be \
delayed. Sorry/thank you ",
      "Thank you for contacting us. We are currently looking into your feedback/enquiry \
and will respond to you within 5 working days. Should you require urgent \
assistance, please contact our hotline at 6213 8008 from Mondays to Fridays, 9 \
am to 5. 30 pm, and Saturdays, 9 am to 12. 30 pm ",
      "I am currently on leave will return back on 31 Dec 2019 and I have limited email \
access will reply you as soon as possible ",
      "I am on oversea project and will be back to work on 16 Dec 2019. For urgent \
matter, you still can try to call my mobile or whatsapp me. Otherwise you can \
call my office at __PHONENUMBER__ for any assistance ",
      "I am currently On Leave, for urgent matter please Call or  Email NTUC Help desk. \
Thank you ",
      "I’m on leave till 21 Jan. Will reply to your email when I return to office.  \
Thanks ",
      "I am on leave till 5 Jan and will reply you once I am back. Have a great day \
ahead",
      "I am overseas holiday from 9 Dec to 17 Dec and not able to reply your email \
soon. For Dr. Kwong Yuk Wah appointment, please email her directly if urgent. \
Otherwise,  wait for my reply when I back to office ",
      "I am out of office. I will get back to you as soon as I can ",
      "I will be away from 16 to 21 Dec 19, will be back to office on 23 Dec 19 ",
      "I am out of office from 23 to 29 Dec. Response to email might be. delay. Pls \
contact me @ my mobile for any urgent matters ",
      "I am out of office and will be back on 29 Jan. For system related matters, \
please contact UXI Team at __EMAILADDRESS__ For technical issues on UCEM, please \
email to NTUC Service Desk __EMAILADDRESS__ and UCEM Support Thanks Connie ",
      "Happy New Year. I'm on leave till 3 Jan 2020. If you are in urgent need of \
assistance regarding digital media matters, please contact Liyin at \
__EMAILADDRESS__ or Nicholas at __EMAILADDRESS__ ",
      "I am out of office today. For UXI related matters, please email to UXI \
__EMAILADDRESS__ For technical issues on U-CEM, please email to NTUC Service \
Desk __EMAILADDRESS__ Best Regards, Hui Lian ",
      "Thank you for your message. I am currently overseas and have limited email \
access, thus my response may be delayed. I will respond to your email when I'm \
back on 23 December 2019. For assistance in Facilities matters, please send your \
request to __EMAILADDRESS__ You may also contact:. Ahmad :  __PHONENUMBER__. Andy: \
__PHONENUMBER__ ",
      "Thanks for your email. I am out of office and will be back next Monday ",
      "Please note that case INC000004579618 is at 70% milestone for the resolution \
time. IT Project / Service  NTUC    Number  INC000004579618  Summary   Unable to outlook email with UDI dtankh to backup email before \
the migration at 12nn    Location        MBGC    Opened          2019 - 12 - 19. \
**** DISCLAIMER ***** ",
      "Please see the attached updated Excel file. Thanks a lot ",
      "Thank you for your email. Your request is being served by :",
      "Please kindly contact Service desk for all IT related matters. Thank you ",
      "Hi, I’m on leave and will be back on mon ", "FYI&NA (if applicable) ",
      "If you need to check Sylvia's calendar urgently, please call Imah at tel no. \
__PHONENUMBER__. Thanks ",
      "I am out of office till 20 Dec and will revert to you when I am back. Have a \
nice day ",
      "This is the mail delivery agent at Symantec Email Security. cloud. I was unable \
to deliver your message to the following addresses: ",
      "Hi, I am on medical leave today",
      "Please click here for instructions on how to register \
in UAM Password Station",
      "I am on annual leave and may have limited access to email. For urgent matters, \
you may contact my colleagues:. 1. For all marketing and campaign matters, \
please contact Carol at __EMAILADDRESS__. 2. For NTUC Family Membership matters, \
please contact Anson at __EMAILADDRESS__. Thanks and have a lovely Christimas \
ahead ",
      "Thank you for your email. I'm on currently away and will be back on 24 Dec 2019. \
I will have limited internet access. For any urgent NTWU matters, please contact \
NTWU at __PHONENUMBER__. Thanks ",
      "I'm on leave on 26 Dec. Please expect delay in my reponse. For urgent matters, \
please reach me at my mobile. Thank you ",
      "I am currently away with limited or no email access. Will return on 6 Jan (Mon). \
Thanks ",
      "Meanwhile, can you confirm where is the delay based on the time stamp. I have  \
asked Eng Yao to send the attachment directly to O365 Email address ",
      "Please be informed that we will be performing urgent network maintenance and the \
details are as follows:-. Maintenance Date and Time:. -          21st Jan 2020 \
(Tuesday) between 7:00pm to 8:00pm. Services will be interrupted:. -         All \
Network and Internet access at OMB 13th floor will be affected. Affected users:. \
-         Staffs located at OMB 13th floor. Sorry for all the inconvenience \
caused ", "Sorry I missed your email. I am on medical leave ",
      "Thank you for your message. I am away this morning and will get back to you once \
back office ", "Away for offsite meeting, back to office on 18 Dec ",
      "Apologies, I am away from office and will be back on 12 December 2019 ",
      "Done, please verify. Thank You ",
      "Please see the updated Excel. Thanks a lot ",
      "The below emails has been removed from “NTUC-I-Messaging-Support” group ",
      "Dear Sender, I am away and will revert to your email on 16 Dec "
  ]
  getParaphrases(8, input_sentences)
"""

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
"""