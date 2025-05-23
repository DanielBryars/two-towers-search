import json

import numpy as np


def parse__passages_json(json_docs:str):  
    json_data = json.loads(json_docs)

    is_selected = np.array(json_data['is_selected'])
    passage_text = np.array(json_data['passage_text'])
    
    # Get indices where is_selected is 1 (selected) and 0 (not selected)
    selected_indices = np.where(is_selected == 1)[0]
    not_selected_indices = np.where(is_selected == 0)[0]
    
    # Create arrays of selected and non-selected passages
    selected_passages = passage_text[selected_indices]
    not_selected_passages = passage_text[not_selected_indices]
    
    return selected_passages, not_selected_passages


if __name__ == '__main__':

    json_docs = '''{ 
    "is_selected": [ 0, 0, 0, 0, 0, 0, 0, 0, 1 ], 
    "passage_text": [ 
        "1 For naproxen controlled-release tablet (e.g., Naprelan®) dosage form: 2 For rheumatoid arthritis, osteoarthritis, and ankylosing spondylitis: 3 Adults—At first, 750 milligrams (mg) (taken as one 750 mg or two 375 mg tablets) or 1000 mg (taken as two 500 mg tablets) once a day. 1 Adults—550 milligrams (mg) for the first dose, then 550 mg every 12 hours or 275 mg every 6 to 8 hours as needed. 2 Your doctor may increase the dose, if necessary, up to a total of 1375 mg per day. 3 Children—Use and dose must be determined by your doctor.", "If you are using over-the-counter naproxen sodium products (such as Aleve), you should follow the instructions on the label. Do not exceed the recommended over-the-counter doses (one tablet or 220 mg, twice a day), and do not take naproxen sodium for more than ten days, unless your healthcare provider recommends it. The recommended starting dosage of naproxen sodium for most people with rheumatoid arthritis, osteoarthritis, or ankylosing spondylitis is 275 mg or 550 mg, twice a day.", 
        "The right Naproxen dosage. Naproxen dosage recommendations for treating pain in osteoarthritis, vary between 500 mg and 1000 mg daily. The maximum Naproxen dosage per day is 1500 mg. In order to avoid any stomach upset is better to take your daily dosage of Naproxen after meals or with a glass of milk. The best dosage for arthritis is the same as in case of osteoarthritis, and it ranges between 500 mg to", 
        "For Adults, the minimum is 275.0 mg, maximum is 1650.0 mg, Pediatric Dosages are, minimum 5.0 mg/kg, maximum = 21.0 mg/kg The safety and efficacy of this drug has not been established in children 2 years or younger. The dosage for adults assumes this maximum will not be exceeded within 24 hours of the maximum dosage. Generally, this means that adults should not take more than three capsules total within 24 hours of the first dose. It doesn\'t matter how you obtain the ibuprofen, OTC or prescription, the maximum is the same.1200 is safe, more should be monitered, like by a doctor, maybe.Actually, no one should be taking 2000mg of ibuprofen. The OTC dosage is 400mg, and the prescription dosage is 800mg.",
        "The recommended nonprescription dosage for most people over age 12 is naproxen sodium 220 mg taken by mouth every 8 to 12 hours as needed. The daily dose should not exceed 660 mg for people under 65 years of age. If you are over age 65, the recommended dose is naproxen sodium 220 mg taken by mouth every 12 hours as needed. The daily dose should not exceed 440 mg for people over 65 years of age.", 
        "Naproxen Dosage. What is the right Naproxen dosage is a very common question for everyone using this medicine. Naproxen can be found on the market under different brand names and in several forms, and is available as 250 mg, 375 mg or 500 mg tablets. The maximum Naproxen dosage per day is 1500 mg. In order to avoid any stomach upset is better to take your daily dosage of Naproxen after meals or with a glass of milk.", 
        "Dose of Naproxen Sodium for Acute Pain For acute pain relief or treatment of painful menstrual periods, the usual naproxen sodium dosing for an adult (over age 18) is 220 mg to 550 mg, twice a day. The maximum recommended daily dosage is 1100 mg per day. The recommended starting dosage of naproxen sodium for most people with rheumatoid arthritis, osteoarthritis, or ankylosing spondylitis is 275 mg or", 
        "Aleve (® Bayer Healthcare) is an over-the-counter (OTC) brand of the drug Naproxen Sodium and is classified as a Non-Steroidal,Anti-Inflammatory Drug or NSAID. According to the manufacturer\'s recommendations, the minimum and maximum oral dosages are as follows... It doesn\'t matter how you obtain the ibuprofen, OTC or prescription, the maximum is the same.1200 is safe, more should be monitered, like by a doctor, maybe.Actually, no one should be taking 2000mg of ibuprofen. The OTC dosage is 400mg, and the prescription dosage is 800mg.", 
        "You should take one tablet every 8 to 10 hours until symptoms abate, although for the first dose you may take two tablets. The maximum adult dose is two tablets within an 8 to 10 hour period and three tablets within a 24-hour period. Aleve should not be taken for longer than 10 days unless recommended by your doctor. You should only give Aleve to a child under 12 years old if recommended by a doctor. " 
        ], 
    "url": [ "http://www.mayoclinic.org/drugs-supplements/naproxen-oral-route/proper-use/DRG-20069820", "http://arthritis.emedtv.com/naproxen-sodium/naproxen-sodium-dosage.html", "http://www.nsaidslist.com/naproxen-dosage/", "http://www.answers.com/Q/What_is_the_maximum_daily_dosage_of_Aleve", "http://arthritis.emedtv.com/naproxen/naproxen-dosage.html", "http://www.nsaidslist.com/naproxen-dosage/", "http://arthritis.emedtv.com/naproxen-sodium/naproxen-sodium-dosage.html", "http://www.answers.com/Q/What_is_the_maximum_daily_dosage_of_Aleve", "https://www.sharecare.com/health/nonsteroidal-anti-inflammatory-drugs/what-maximum-daily-dosage-aleve" ] }'''
    query = 'aleve maximum dose'

    selected_passages, not_selected_passages = parse__passages_json(json_docs)

    print(selected_passages)
    print(f"not selected count: {len(not_selected_passages)}")

