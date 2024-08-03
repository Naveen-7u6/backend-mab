import requests

def getAuth():
    url = "https://xmloutapi.tboair.com/api/v1/Authenticate/ValidateAgency"

    payload = {
        "UserName": "Onttest",
        "Password":"Ot@131020",
        "BookingMode":"API",
        "IPAddress":"192.16910. 22"
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    auth_token = response["TokenId"]

    return auth_token


def getFlightDetails(origin, destination):

    auth = getAuth()

    flightDetailsURL = "https://xmloutapi.tboair.com/API/V1/Search/Search"

    payload = {
        {
            "IPAddress": "192.168.11.92",
            "TokenId":auth,
            "EndUserBrowserAgent":"Mozilla/5.0(Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, likeGecko) Chrome/70.0.3538.110 Safari/537.36",
            "PointOfSale":"ID",
            "RequestOrigin":"India",
            "UserData": None,
            "JourneyType": 1,
            "AdultCount": 1,
            "ChildCount": 0,
            "InfantCount": 0,
            "FlightCabinClass": 1,
            "Segment":
                [
                    {
                     "Origin": origin,
                     "Destination": destination,
                     "PreferredDepartureTime": "",
                     "PreferredArrivalTime": "",
                     "PreferredAirlines": []
                    }
            ]
    }

    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(flightDetailsURL, json=payload, headers=headers)

    print(response)


    return response
    