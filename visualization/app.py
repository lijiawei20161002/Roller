from flask import Flask, render_template, request, jsonify, Response  
  
app = Flask(__name__)  
  
@app.route("/") #, methods=["GET", "POST"])  
def index(): 
    ''' 
    if request.method == "POST":  
        num_jobs = int(request.form["num_jobs"])  
        num_servers = int(request.form["num_servers"])  
        service_rates = list(map(float, request.form["service_rates"].split(',')))  
        max_queue_length = int(request.form["max_queue_length"])  
        rl_algorithms = request.form["rl_algorithms"].split(',')  
  
        # Call your queueing system function and pass the input parameters  
        # results = your_queueing_system_function(num_jobs, num_servers, service_rates, max_queue_length, rl_algorithms)  
  
        # Dummy results for demonstration  
        results = {  
            "cumulative_rewards": [  
                [0, 10, 25, 45, 60, 80, 110, 130, 150, 170],  
                [0, 8, 22, 38, 55, 75, 100, 120, 140, 160],  
                [0, 12, 28, 47, 65, 85, 115, 135, 155, 175]  
            ]  
        }  
  
        return jsonify(results)  
    '''
  
    html_content = render_template('index.html')  
    return Response(html_content, content_type='text/html; charset=utf-8')  
  
if __name__ == "__main__":  
    app.run(debug=True)  
