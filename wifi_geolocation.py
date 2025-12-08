import subprocess
import re
import requests
import json
import sys

# API Keys provided by user
API_KEYS = [
    "AIzaSyCMT1JkSjfY1xbgPyaQCQHN3ET1K8TZW9c"
]

def get_wifi_networks():
    """
    Scans for available Wi-Fi networks using the Windows 'netsh' command.
    Returns a list of dictionary objects suitable for the Google Geolocation API.
    """
    networks = []
    try:
        # Run the netsh command to show networks with BSSID
        result = subprocess.check_output(
            ['netsh.exe', 'wlan', 'show', 'networks', 'mode=bssid'],
            universal_newlines=True,
            encoding='utf-8', # Attempt utf-8 first
            errors='ignore'   # Ignore encoding errors if utf-8 fails slightly
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing netsh: {e}")
        return []

    # Regex to parse the output
    # Typical block:
    # SSID 1 : NetworkName
    # ...
    #     BSSID 1                 : 00:11:22:33:44:55
    #          Signal             : 80%
    #          Radio type         : 802.11ac
    #          ...
    
    # We need to capture blocks associated with each BSSID
    # Since netsh output structure is a bit nested (SSID can have multiple BSSIDs),
    # we'll iterate line by line to maintain context.
    
    current_bssid = None
    current_signal = None
    current_channel = None
    
    lines = result.splitlines()
    for line in lines:
        line = line.strip()
        
        # Match BSSID (MAC address)
        bssid_match = re.search(r'BSSID \d+\s*:\s*([0-9a-fA-F:]{17})', line)
        if bssid_match:
            # If we were tracking a previous BSSID, store it
            if current_bssid and current_signal:
                # Convert signal % to dBm (APPROXIMATION)
                # 100% ~= -50dBm, 0% ~= -100dBm. formula: (percent / 2) - 100
                signal_dbm = (int(current_signal) / 2) - 100
                networks.append({
                    "macAddress": current_bssid,
                    "signalStrength": int(signal_dbm),
                    "channel": int(current_channel) if current_channel else None
                })
            
            # Reset for new BSSID
            current_bssid = bssid_match.group(1)
            current_signal = None
            current_channel = None
            continue

        # Match Signal
        signal_match = re.search(r'Signal\s*:\s*(\d+)%', line)
        if signal_match:
            current_signal = signal_match.group(1)
            continue
            
        # Match Channel
        channel_match = re.search(r'Channel\s*:\s*(\d+)', line)
        if channel_match:
            current_channel = channel_match.group(1)
            continue

    # Append the last found network
    if current_bssid and current_signal:
        signal_dbm = (int(current_signal) / 2) - 100
        networks.append({
            "macAddress": current_bssid,
            "signalStrength": int(signal_dbm),
            "channel": int(current_channel) if current_channel else None
        })

    return networks

def get_geolocation(wifi_access_points, api_key):
    """
    Sends the Wi-Fi access points to Google Geolocation API to get lat/long.
    """
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    
    payload = {
        "considerIp": "true", # Use IP as fallback or aid
        "wifiAccessPoints": wifi_access_points
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed with key ending in ...{api_key[-4:]}: {e}")
        if response is not None:
             print(f"Response content: {response.text}")
        return None

def get_device_location():
    """
    Scans for valid Wi-Fi networks and attempts to retrieve location using available API keys.
    Returns:
        tuple: (latitude, longitude, accuracy) or (None, None, None)
    """
    print("Scanning for Wi-Fi networks...")
    wifi_list = get_wifi_networks()
    
    if not wifi_list:
        print("No Wi-Fi networks found.")
        return None, None, None

    # Filter out valid entries only (MAC and Signal are strict requirements usually)
    valid_wifi = [w for w in wifi_list if w.get('macAddress') and w.get('signalStrength')]
    
    if not valid_wifi:
        print("No valid Wi-Fi data parsed.")
        return None, None, None

    # Try keys
    location_data = None
    for key in API_KEYS:
        # print(f"Trying API Key: ...{key[-5:]}")
        location_data = get_geolocation(valid_wifi, key)
        if location_data:
            break

    if location_data:
        location = location_data.get('location', {})
        lat = location.get('lat')
        lng = location.get('lng')
        accuracy = location_data.get('accuracy')
        return lat, lng, accuracy
    
    return None, None, None

def main():
    lat, lng, accuracy = get_device_location()

    if lat and lng:
        print("\n--- RESULTS ---")
        print(f"Latitude:  {lat}")
        print(f"Longitude: {lng}")
        print(f"Accuracy:  {accuracy} meters")
        
        # Optional: Generate Google Maps Link
        print(f"Google Maps: https://www.google.com/maps/search/?api=1&query={lat},{lng}")
    else:
        print("Could not obtain location.")

if __name__ == "__main__":
    main()
