$(document).ready(function(){
    //connect to the socket server.
	
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
	var samples = [];
	var results;
	var results_obtained = false;
	var results_obtained_italian_flag = false;
  var results_obtained_sa = false;
	var status = ''
	i = 0;

	var key = "";                // API KEY
	var id = ""; // CSE ID
	var q = "cats";                        // QUERY

	// var positive = "#29CB1C";
	// var neutral = "#F8F8F8";
	// var negative = "#CB1C29";

	//Colour scheme for d3 svgs

	var positive = "#1EB619";
	var neutral = "#F8F8F8";
	var negative = "#B6191E";





	var results = document.getElementById("results");
	var loading = document.getElementById("loading");
	var buttons = document.getElementById("button_wrapper")
	var sentiment = document.getElementById("sentiment");
	var topic_analysis = document.getElementById("topic_analysis")
	var summarisation = document.getElementById("s_cont");
	var bar_chart = document.getElementById("bar_chart");
  var bar_chart3 = document.getElementById("bar_chart3");
	var sum_load = document.getElementById("sum_load");
	var loading_text = document.getElementById("loading_text");
	var sent_cont = document.getElementById("sent_cont")
	var topic_cont = document.getElementById("topic_cont")
	var bar_chart_two = document.getElementById("bar_chart_two");
	var load_wrapper = document.getElementById("load_wrapper")

	function triggersearch(query){

		var JSElement = document.createElement('script');
		JSElement.src = `https://www.googleapis.com/customsearch/v1?key=${key}&cx=${id}&q=${query}` + '&searchType=image&imgSize=huge&rights=cc_publicdomain&callback=hndlr';
		document.getElementsByTagName('head')[0].appendChild(JSElement);
	}


	results.style.display = 'none';
	buttons.style.display = 'none';
	sentiment.style.display = 'none';
	topic_analysis.style.display = 'none';
	summarisation.style.display = 'none';
	bar_chart.style.display = 'none'; // check redundant
	bar_chart_two.style.display = 'none';


	sum_load.style.display = 'none';
	sent_cont.style.display = 'none';
	topic_cont.style.display = 'none';
	$('#loading').html('Loading...');

	var intervalID = window.setInterval(updatePage, 3000);

	function updatePage() {
		// go check API
		if (results_obtained) {
			if (status !== 'Finished') {
				// background.style.display = 'block';
				results.style.display = 'block';
				buttons.style.display = 'block';
				loading.style.display = 'none';
				loading_text.style.display = 'none';
				load_wrapper.style.display = 'none';
				buttons.style.display = 'block';
				sentiment.style.display = 'block';
				topic_analysis.style.display = 'block';
				summarisation.style.display = 'none';
				bar_chart.style.display = 'block';// check if redundant
        bar_chart3.style.display = 'block';// check if redundant
				bar_chart_two.style.display = 'block';

				sum_load.style.display = 'block';
				sent_cont.style.display = 'block';
				topic_cont.style.display = 'block';
			}
			else {
				document.getElementById("summ_link").href="#s_cont";
				sum_load.style.display = 'none';
				summarisation.style.display = 'block';
			}
			// clearInterval(intervalID);
		}
		else {
			loading_text.style.display = 'none';
			sentiment.style.display = 'none';
			topic_analysis.style.display = 'none';
			summarisation.style.display = 'none';
			bar_chart.style.display = 'none'; //check
      bar_chart3.style.display = 'none';
			bar_chart_two.style.display = 'none';
			sum_load.style.display = 'none';
			load_text = '<p>' + samples[i] + '</p>';
			if (load_text == undefined) {
				load_text = "Please wait until previous analysis is complete";
			}
			$('#loading_text').fadeIn("slow").html(load_text);
			$('#loading').html(status);

			i++;
		}


	}

	socket.on('keyword', function(msg) {
		query = msg.keyword;
		triggersearch(query);
	});

    //receive details from server
    socket.on('loading', function(msg) {
		console.log("Received tweet" + msg.tweet);

		samples = msg.tweet
		console.log(samples)

        // numbers_string = '<h1> Loading: ' + msg.message + '</h1>' +  '<p>' + msg.tweet.toString() + '</p>';

		// $('#log').fadeIn("slow").html(numbers_string);


	});

	socket.on('status', function(msg) {
		status = msg.status.toString()
	});

    socket.on('results', function(msg) {
        console.log("Received SA results");

        topics = msg.topic

		sentiment_string = '<p>' + msg.sentiment.toString() + '</p>';

		console.log(sentiment_string);

		summary = msg.summary

        output_string = '';
        console.log(topics);
        for (var i = 0; i < topics.length; i++){
            output_string = output_string + '<p>' + topics[i].toString() + '</p>';
		}

		summary_string = '';
		for (var i = 0; i < summary.length; i++){
			summary_string = summary_string + '<li>' + summary[i].toString() + '</li>';
		}

		summary_string = '<ul>' + summary_string + '</ul>';

		topic_string = output_string;
		topic_string = '';
		sentiment_string = '';

		$('#sentiment').html(sentiment_string);
		$('#topic_analysis').html(topic_string);


    document.getElementById('bar_chart3').addEventListener('hover', function(e) {
        e.currentTarget.setAttribute('fill', 'black');
    });



    if (results_obtained_sa == false) {
      d3.json('/get-data-sa-json', function(json){
			  data = json;
			  // data = JSON.parse(data);
        // console.log('THIS IS data' + data)
        d3.json('/get-data-sa-idx', function(idx){
          idx = idx;

          var sentiment_emoji = {'positive':'😊', 'neutral':'😐', 'negative':'😡'};
          var sentiment_colour = {'positive':positive, 'neutral':neutral, 'negative':negative};

          var total_length = 0
          for (var i = 0; i < idx.length; i++){
            total_length = total_length + data[idx[i]][2].length
          }

          var y_sa = 10
          var line_width = 25.5;
          var height_b = total_length * line_width + y_sa + 30 * idx.length;
          var width_b = 800;

		  var svg4 = d3.select('#bar_chart3').classed("svg-container", true).append("svg")
		  	
              .attr("width", width_b)
			  .attr("height", height_b)
			.classed("svg-content-responsive", true)
			  .attr("preserveAspectRatio", "xMinYMin meet") 
			//   .attr("viewBox", "0 0 "+ height_b + " " +  width_b)
              .style('pointer-events', 'all');



          for (var i = 0; i < idx.length; i++){

            var n_lines = data[idx[i]][2].length
            console.log('NUMBER OF LINES ' + i + ' ' + n_lines)
            var height_sa = line_width * n_lines;

            var twitter_size = 60;
            var img_t = svg4.append('foreignObject')
                .attr('x', 0)
                .attr('y', y_sa + height_sa/2 - twitter_size/2)
                .attr("width", twitter_size)
                .attr("height", twitter_size)

            var imgg = img_t.append('xhtml:img')
                // .append('img')
                .attr('src', "static/twitter_logo.png")
                .attr('width', twitter_size)
                .attr('height', twitter_size);

            var rectangle = svg4.append("rect")
                .attr('x', 100)
                .attr('y', y_sa - 5)
                .attr("width", 600)
                .attr("height", line_width * n_lines + 10)
                // 27
                .attr("rx", 4)
                .style("fill", function(d) { return '#00acee'; })
                .style("stroke", function(d) { return d3.rgb('#00acee').darker(); })
                // .style("fill", function(d) { return sentiment_colour[data[i][1]]; })
                // .style("stroke", function(d) { return d3.rgb(sentiment_colour[data[i][1]]).darker(); })
                .attr("fill-opacity","1")
                .style("pointer-events", "all")


            var fbox = svg4.append("foreignObject")
                .attr('x', 100)
                .attr('y', y_sa)
                .attr("width", 600)
                .attr("height", height_sa)

            var fboxleft = svg4.append("foreignObject")
                .attr('x', 0)
                .attr('y', y_sa + height_sa/2 - 13)
                .attr("width", twitter_size)
                .attr("height", height_sa)

            var fboxright = svg4.append("foreignObject")
                .attr('x', 700)
                .attr('y', y_sa + height_sa/2 - 17)
                .attr("width", 100)
                .attr("height", height_sa + 2)

            svg4.on("mouseover", function(d){
                d3.select(event.currentTarget)
                .style("fill", "black");
                })
                .on("mouseout", function(d){
                d3.select(event.currentTarget)
                .style("fill", "black");
                });

            for (var j = 0; j < data[idx[i]][2].length; j++){
                var tbox = fbox.append("xhtml:div")
                    .style("font", "21px 'Helvetica Neue'")
                    .html(data[idx[i]][2][j]);
            };

            var tboxleft = fboxleft.append("xhtml:div")
                .style("font", "16px 'Helvetica Neue'")
                .html(i+1);
                // .html(data[i][0]);

            var tboxright = fboxright.append("xhtml:div")
                .style("font", "25px 'Helvetica Neue'")
                .html(sentiment_emoji[data[idx[i]][1]]);

            var click_rect = svg4.append("rect")
                .attr('id', 'hov' + i)
                .attr('x', 100)
                .attr('y', y_sa - 5)
                .attr("width", 600)
                .attr("height", line_width * n_lines + 10)
                .attr("rx", 4)
                .attr("fill", sentiment_colour[data[idx[i]][1]])
                .attr('stroke', sentiment_colour[data[idx[i]][1]])
                .attr("opacity", 0)
                .style("pointer-events", "all")
                // .attr('class', 'click-capture')

            document.getElementById('hov' + i)
              .addEventListener('mouseover', function(e) {
              e.currentTarget.setAttribute('opacity', '.7');
              // e.getElementById('em' + i).setAttribute('opacity', '0');
            });

            document.getElementById('hov' + i)
              .addEventListener('mouseout', function(e) {
              e.currentTarget.setAttribute('opacity', '0');
              // document.getElementById('em' + i).setAttribute('opacity', '1');
            });

            y_sa = y_sa + height_sa + 30
            }


        });

      });
      results_obtained_sa = true;
    };


		// insert TA plot
		var img = document.createElement("IMG");
		img.style.width = "70vw"
		img.width = "850";
		img.src = "static/wordcloud.png";
		document.getElementById('topic_analysis').appendChild(img);

		if (results_obtained_italian_flag == false) { //prevents frontend from creating multiple plots
			// insert TA plot
			var img3 = document.createElement("IMG");
			img3.style.width = "70vw"
			img3.width = "900";
			img3.src = "static/TA_chart.png";
			document.getElementById('topic_analysis2').appendChild(img3);

		}


		if (results_obtained_italian_flag == false) {

			var svgWidth = 800;
			var svgHeight = 300;
			var barPadding = 5;
			var labels = ["positive", "neutral", "negative"];
			var colors = [positive, neutral, negative];
			var indices = [0, 1, 2];
			var datasettwo;

			d3.json('/get-data-italian-flag', function(json){
			  datasettwo = json;
			  datasettwo = JSON.parse(datasettwo);

			  var barWidth = (svgWidth / datasettwo.length);

			  var svgtwo = d3.select("#bar_chart_two")
			  		.classed("svg-container", true)
			  		.append("svg")
				  	.attr("width", svgWidth)
				  	.attr("height", svgHeight)
				  	.classed("svg-content-responsive", true)
					.attr("preserveAspectRatio", "xMinYMin meet") 
					// .attr("viewBox", "0 0 "+ svgHeight + " " +  svgWidth)

			  var barChart = svgtwo.selectAll('rect')
				.data(indices)
				.enter()
				.append('rect')
				.attr("y", function(d) {
				  return svgHeight - datasettwo[d] * svgHeight;
				})
				.attr("height", function(d) {
				  return datasettwo[d] * svgHeight;
				})
				.attr("width", barWidth - barPadding)
				.attr("transform", function(d, i) {
				  var translate = [barWidth * i, 0];
				  return "translate("+ translate + ")";
				})
				.attr("fill", function(d) {
				  return colors[d]
				});

				var text = svgtwo.selectAll("text")
				  .data(indices)
				  .enter()
				  .append("text")
				  .text(function(d) {
					return (datasettwo[d] * 100).toString().substring(0, 4) + "% " + labels[d]
				  })
				  .attr("y", function(d, i) {
					return svgHeight - datasettwo[d] * svgHeight - 2;
				  })
				  .attr("x", function(d, i) {
					return barWidth * i;
				  })
				  .attr("fill", function(d) {
					return colors[d]
				  });

				var text = svgtwo.selectAll("text")
				  .data(indices)
				  .enter()
				  .append("text")
				  .text(function(d) {
					return "l"
				  })
				  .attr("y", function(d, i) {
					return svgHeight - datasettwo[d]- 2;
				  })
				  .attr("x", function(d, i) {
					return barWidth * i;
				  })
				  .attr("fill", "#steelblue");

			})


				  results_obtained_italian_flag = true;

			// })
		  }

		$('#Summarisation').html(summary_string)


		if (results_obtained == false) {

			var w = 900;                        //width
			var h = 600;                        //height
			var padding = {top: 50, right: 30, bottom: 50, left:80};
			var dataset;
			//Set up stack method
			var stack = d3.layout.stack();


			d3.json('/get-data', function(json){
				dataset = json;

				//Data, stacked
				stack(dataset);

				// var color_hash = {
				// 		0 : ["Postive","#1f77b4"],
				// 		1 : ["Neutral","#2ca02c"],
				// 		2 : ["Negative","#ff7f0e"]
        //
				// };
        var color_hash = {
						0 : ["Postive",positive],
						1 : ["Neutral",neutral],
						2 : ["Negative",negative]

				};


				//Set up scales
        var xbin = (new Date(dataset[0][1].time) - new Date(dataset[0][0].time)) / 1000;
        var xmax = d3.time.second.offset(new Date(dataset[0][dataset[0].length-1].time), xbin);

        if (xbin / 3600 > 10) {
          var xaxis_time = d3.time.days
          var xaxis_interval = 1
        }
        else {
          if (xbin / 60 > 100) {
            var xaxis_time = d3.time.hours
            var xaxis_interval = 6
          }
          else {
            if (xbin / 60 > 20) {
              var xaxis_time = d3.time.minutes
              var xaxis_interval = 15
            }
            else {
              if (xbin > 100) {
                var xaxis_time = d3.time.minutes
                var xaxis_interval = 5
              }
              else {
                var xaxis_time = d3.time.minutes
                var xaxis_interval = 1
              };
            };
        };
      };

        console.log('CHECK THESE OUTTTTTTT why2 ' + xaxis_interval)

				var xScale = d3.time.scale()
					.domain([new Date(dataset[0][0].time), xmax])
					.rangeRound([0, w-padding.left-padding.right]);

				var yScale = d3.scale.linear()
					.domain([0,
						d3.max(dataset, function(d) {
							return d3.max(d, function(d) {
								return (d.y0 + d.y)*1.15;
							});
						})
					])
					.range([h-padding.bottom-padding.top,0]);

				var xAxis = d3.svg.axis()
							.scale(xScale)
							.orient("bottom")
              .ticks(8);
							// .ticks(xaxis_time, xaxis_interval);

				var yAxis = d3.svg.axis()
							.scale(yScale)
							.orient("left")
							.ticks(10);



				//Easy colors accessible via a 10-step ordinal scale
				var colors = d3.scale.category10();

				//Create SVG element
				var svg = d3.select("#bar_chart")
							.classed("svg-container", true)
							.append("svg")
							.attr("width", w)
							.attr("height", h)
							.classed("svg-content-responsive", true)
							.attr("preserveAspectRatio", "xMinYMin meet") 
							// .attr("viewBox", "0 0 "+ h + " " + w);

				// Add a group for each row of data
				var groups = svg.selectAll("g")
					.data(dataset)
					.enter()
					.append("g")
					.attr("class","rgroups")
					.attr("transform","translate("+ padding.left + "," + (h - padding.bottom) +")")
					.style("fill", function(d, i) {
						return color_hash[dataset.indexOf(d)][1];
					});

				// Add a rect for each data value
				var rects = groups.selectAll("rect")
					.data(function(d) { return d; })
					.enter()
					.append("rect")
					.attr("width", 2)
					.style("fill-opacity",1e-6);


				rects.transition()
					.duration(function(d,i){
						return 500 * i;
					})
					.ease("linear")
					.attr("x", function(d) {
						return xScale(new Date(d.time));
					})
					.attr("y", function(d) {
						return -(- yScale(d.y0) - yScale(d.y) + (h - padding.top - padding.bottom)*2);
					})
					.attr("height", function(d) {
						return -yScale(d.y) + (h - padding.top - padding.bottom);
					})
					.attr("width", 15)
					.style("fill-opacity",1);

					var axis1 = svg.append("g")
						.attr("class","x axis")
						.attr("transform","translate(60," + (h - padding.bottom) + ")")
						.call(xAxis);

          axis1.selectAll("line")
            .style("stroke", "white")

          axis1.selectAll("path")
            .style("stroke", "white")

          axis1.selectAll("text")
            .style("stroke", "white")
            .style("fill", "white");

					var axis2 = svg.append("g")
						.attr("class","y axis")
						.attr("transform","translate(" + (padding.left-20) + "," + padding.top + ")")
						.call(yAxis);

          axis2.selectAll("line")
            .style("stroke", "white")

          axis2.selectAll("path")
            .style("stroke", "white")

          axis2.selectAll("text")
            .style("stroke", "white")
            .style("fill", "white");

					// adding legend

					var legend = svg.append("g")
									.attr("class","legend")
									.attr("x", w - padding.right - 65)
									.attr("y", 25)
									.attr("height", 100)
									.attr("width",100);

					legend.selectAll("g").data(dataset)
						.enter()
						.append('g')
						.each(function(d,i){
							var g = d3.select(this);
							g.append("rect")
								.attr("x", w - padding.right - 125)
								.attr("y", i*35 + 5 + 10)
								.attr("width", 10)
								.attr("height",10)
								.style("fill",color_hash[String(i)][1]);

							g.append("text")
							.attr("x", w - padding.right - 110)
							.attr("y", i*35 + 20 + 10)
							.attr("height",30)
							.attr("width",100)
							.style("fill",color_hash[String(i)][1])
							.text(color_hash[String(i)][0])
              .attr('font-size', 20);
						});

					svg.append("text")
					.attr("transform","rotate(-90)")
					.attr("y", 0 )
					.attr("x", 0-(h/2) - 100)
					.attr("dy","1em")
					.text("Number of Tweets")
          .style("fill","white");

				svg.append("text")
				.attr("class","xtext")
				.attr("x",w/2 - padding.left + 80)
				.attr("y",h - 5)
				.attr("text-anchor","middle")
				.text("Time")
        .style("fill","white");


			});

			results_obtained = true;

		}


    });

});
