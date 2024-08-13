import librosa
import numpy as np
import concurrent.futures
from flask import Flask, request, Response
from flask_cors import CORS

def calculate_overall_tempo(y, sr):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    if len(onset_times) < 2:
        raise ValueError("Not enough onsets detected to calculate tempo.")
    
    intervals = np.diff(onset_times)
    avg_interval = np.mean(intervals)
    tempo_bpm = 60 / avg_interval
    
    return tempo_bpm

def cut_notes(notes):
    return [note for i, note in enumerate(notes) if i == 0 or note[0] != notes[i-1][0]]

def get_dur(avgs, duration):
    durs = []
    notes = [avg[0] for avg in avgs]
    times = [avg[2] for avg in avgs]
    for i, note in enumerate(notes):
        if i < len(times)-1:
            durs.append(times[i+1]-times[i])
        else:
            durs.append(duration-times[i])

    for i in range(len(durs)):
        avgs[i].append(durs[i])

    return avgs

def get_tempo(avgs, duration, bps1, bps2):
    durs = []
    rat = bps1/bps2
    notes = [avg[0] for avg in avgs]
    times = [avg[2] for avg in avgs]
    for i, note in enumerate(notes):
        if i < len(times)-1:
            durs.append(times[i+1]-times[i])
        else:
            durs.append(duration-times[i])

    for i in range(len(durs)):
        avgs[i].append(rat/(durs[i]))

    return avgs

def compare_durations(original_notes, user_notes, attribute, tolerance=0.5):
    duration_mismatches = []

    for original_note, user_note in zip(original_notes, user_notes):
        if abs(original_note[attribute] - user_note[attribute]) > tolerance:
            duration_mismatches.append((original_note, user_note))

    return duration_mismatches

def cust_round_for_cons(a):
    ra = float(int(a))
    if a - ra >= 0.5 and a - ra < 0.65:
        return a+0.25

    elif a-ra >= 0.65 and a-ra < 0.75:
        return a
    elif a - ra >= 0.75:
        return ra+1
    
    elif a - ra >= 0.4 and a - ra < 0.5:
        return ra+0.5

    elif ra == 10:
        return 10

    else:
        return a+0.25

def diff(a, b):
    return (abs(a - b)/a)*10

def match_notes(note1, note2, attribute_index=1, param=5):
    """
    Compare two notes based on a specific attribute (e.g., pitch, volume).
    By default, it compares based on pitch.
    """
    return abs(note1[attribute_index] - note2[attribute_index]) <= param

def lcs_algorithm(original_notes, user_notes, attribute_index=1, param=5):
    """
    Find the longest common subsequence (LCS) between original and user notes.
    """
    m = len(original_notes)
    n = len(user_notes)
    
    # Create a 2D DP table to store lengths of LCS.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if match_notes(original_notes[i - 1], user_notes[j - 1], attribute_index, param):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # The length of the LCS is dp[m][n]
    lcs_length = dp[m][n]
    
    return lcs_length

def extract_notes(audio_path, original):
    y, sr = librosa.load(audio_path)
    y1, sr1 = librosa.load(original)

    bps1 = calculate_overall_tempo(y1, sr1)
    bps2 = calculate_overall_tempo(y, sr)

    bpm = calculate_overall_tempo(y, sr)
    duration = librosa.get_duration(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C8'))
    avgs = []
    prev_note = ""

    zcr = librosa.feature.zero_crossing_rate(y)

    for t in range(pitches.shape[1]):
        indices = magnitudes[:, t] > 0.01

        if np.any(indices):
            index = np.argmax(magnitudes[indices, t])
            pitch = pitches[indices, t][index]
            note = librosa.hz_to_note(pitch)
            
            # Compute volume for the detected pitch
            
            if pitch > 0 and note != prev_note:
                prev_note = note
                tme = librosa.frames_to_time(t)
                amplitude = magnitudes[index, t]
                zcr_value = zcr[0, t]
                avgs.append([note, pitch, tme, amplitude, zcr_value])

            elif note == prev_note:
                avgs[-1][1] = (avgs[-1][1] + pitch)/2
    avgs = get_dur(avgs, duration)
    avgs = get_tempo(avgs, duration, bps1, bps2)
    return avgs

def extract_notes_concurrently(orig_file, user_file):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_orig = executor.submit(extract_notes, orig_file, orig_file)
        future_user = executor.submit(extract_notes, user_file, orig_file)
        
        orig = future_orig.result()
        user = future_user.result()
        
    return orig, user

def rm_unknowns(avgs1, avgs2):
    notes1 = [note[0] for note in avgs1]
    notes2 = [note[0] for note in avgs2]

    notesr = []
    for i,note in enumerate(notes1):
        if note not in notes2:
            notesr.append(avgs1[i])


    for i,note in enumerate(notes2):
        if note not in notes1:
            notesr.append(avgs2[i])

    return notesr

def comp(original_notes, user_notes, attribute_index=1, param=5):
    """
    Calculate the accuracy of user-played notes compared to original notes.
    """
    lcs_length = lcs_algorithm(original_notes, user_notes, attribute_index, param)
    accuracy = lcs_length / len(original_notes)
    return accuracy*10


def c(original_notes, user_notes):
    notesr = rm_unknowns(original_notes, user_notes)
    notes1 = [note[0] for note in original_notes]
    return diff(len(notes1), len(notesr))

def nd(original_notes, user_notes):   
    A = compare_durations(original_notes, user_notes, 5)
    return cust_round_for_cons((abs(len(user_notes)-len(A))/len(user_notes))*10)

def pa(original_notes, user_notes, param=5):
    return cust_round_for_cons(comp(original_notes, user_notes, param=param))

def tr(original_notes, user_notes, param=0.05):
    return comp(original_notes, user_notes, 2, param)

def d(original_notes, user_notes):
    A = []
    for note2 in user_notes:
        for note1 in original_notes:
            if match_notes(note1, note2, attribute_index=3, param=0):
                A.append(note2)
    A = cut_notes(A)
    matches = len(A)
    total = len(original_notes)
    if matches > total:
        matches = total - abs(matches - total)

    return cust_round_for_cons(matches*10/total)

def a(original_notes, user_notes, staccato_threshold=0.05, legato_threshold=0.05):
    A = []
    for note2 in user_notes:
        for note1 in original_notes:
            if note1[4] > staccato_threshold and note2[4] > staccato_threshold:
                A.append(note2)
            elif note1[4] < legato_threshold and note2[4] < legato_threshold:
                A.append(note2)
    A = cut_notes(A)
    matches = len(A)
    total = len(original_notes)
    if matches > total:
        matches = total - abs(matches - total)
    return (matches/total)*10

def tc(original_notes, user_notes):
    return comp(original_notes, user_notes, 6, param=10)

def main_concurrently(original_notes, user_notes):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(c, original_notes, user_notes): "c",
            executor.submit(nd, original_notes, user_notes): "nd",
            executor.submit(pa, original_notes, user_notes): "pa",
            executor.submit(tr, original_notes, user_notes): "tr",
            executor.submit(d, original_notes, user_notes): "d",
            executor.submit(a, original_notes, user_notes): "a",
            executor.submit(tc, original_notes, user_notes): "tc"
        }
        
        results = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
    
    dats = ["c", "nd", "pa", "tr", "d", "a", "tc"]
    result = []
    for dat in dats:
        value = results[dat]
        result.append(f"{dat}: {value}/10")
    
    return "\n".join(result)

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'original_file' not in request.files or 'user_played_file' not in request.files:
        print(list(request.files))
        return 'Missing files', 400

    original_file = request.files['original_file']
    user_file = request.files['user_played_file']

    # Save the uploaded files to disk
    orig_file_path = "original.wav"
    user_file_path = "user.wav"
    original_file.save(orig_file_path)
    user_file.save(user_file_path)

    # Process the files concurrently
    orig, user = extract_notes_concurrently(orig_file_path, user_file_path)
    result = main_concurrently(orig, user)

    # Send the response back to the client
    return Response(result, content_type="text/plain")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)