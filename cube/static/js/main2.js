var state = [];
var rotateIdxs_old = null;
var rotateIdxs_new = null;
var stateToFE = null;
var FEToState = null;
var legalMoves = null;

var solveStartState = [];
var solveMoves = [];
var solveMoves_rev = [];
var solveIdx = null;
var solution_text = null;

var faceNames = ["top", "bottom", "left", "right", "back", "front"];
var colorMap = {0: "#ffffff", 1: "#ffff1a", 4: "#0000ff", 5: "#33cc33", 2: "#ff8000",3: "#e60000"};
var lastMouseX = 0,
  lastMouseY = 0;
var rotX = -30,
  rotY = -30;

var moves = []

var finalState = []

function reOrderArray(arr,indecies) {
	var temp = []
	for(var i = 0; i < indecies.length; i++) {
		var index = indecies[i]
		temp.push(arr[index])
	}

	return temp;
}

/*
	Rand int between min (inclusive) and max (exclusive)
*/
function randInt(min, max) {
	return Math.floor(Math.random() * (max - min)) + min;
}

function clearCube() {
  for (i = 0; i < faceNames.length; i++) {
    var myNode = document.getElementById(faceNames[i]);
    while (myNode.firstChild) {
      myNode.removeChild(myNode.firstChild);
    }
  }
}

function setStickerColors(newState) {
	state = newState
  clearCube()

  idx = 0
  for (i = 0; i < faceNames.length; i++) {
    for (j = 0; j < 9; j++) {
      var iDiv = document.createElement('div');
      iDiv.className = 'sticker';
      iDiv.style["background-color"] = colorMap[Math.floor(newState[idx]/9)]
      document.getElementById(faceNames[i]).appendChild(iDiv);
      idx = idx + 1
    }
  }
}

function buttonPressed(ev) {
	var face = ''
	var direction = '1'

	if (ev.shiftKey) {
		direction = '-1'
	}
	if (ev.which == 85) {
		face='U'
	} else if (ev.which == 68) {
		face = 'D'
	} else if (ev.which == 76) {
		face = 'L'
	} else if (ev.which == 82) {
		face = 'R'
	} else if (ev.which == 66)  {
		face = 'B'
	} else if (ev.which == 70) {
		face = 'F'
	} else if (ev.which == 77 || ev.which == 109){
		face = 'M'
	}else if (ev.which == 69 || ev.which == 101){
		face = 'E'
	}else if (ev.which == 83 || ev.which == 115){
		face = 'S'
	}else if (ev.which == 88 || ev.which == 120){
		face = 'x'
	}else if (ev.which == 89 || ev.which == 121){
		face = 'y'
	}else if (ev.which == 90 || ev.which == 122){
		face = 'z'
	}else if (ev.which == 117) {
		face='u'
	} else if (ev.which == 100) {
		face = 'd'
	} else if (ev.which == 108) {
		face = 'l'
	} else if (ev.which == 114) {
		face = 'r'
	} else if (ev.which == 98)  {
		face = 'b'
	} else if (ev.which == 102) {
		face = 'f'
	}

	if (face != '') {
		clearSoln();
		moves.push(face + "_" + direction);
		nextState();
	}
}


function enableScroll() {
	document.getElementById("first_state").disabled=false;
	document.getElementById("prev_state").disabled=false;
	document.getElementById("next_state").disabled=false;
	document.getElementById("last_state").disabled=false;
}

function disableScroll() {
	document.getElementById("first_state").blur(); //so keyboard input can work without having to click away from disabled button
	document.getElementById("prev_state").blur();
	document.getElementById("next_state").blur();
	document.getElementById("last_state").blur();

	document.getElementById("first_state").disabled=true;
	document.getElementById("prev_state").disabled=true;
	document.getElementById("next_state").disabled=true;
	document.getElementById("last_state").disabled=true;
}

/*
	Clears solution as well as disables scroll
*/
function clearSoln() {
	solveIdx = 0;
	solveStartState = [];
	solveMoves = [];
	solveMoves_rev = [];
	solution_text = null;
	document.getElementById("solution_text").innerHTML = "Solution:";
	disableScroll();
}

function setSolnText(setColor=true) {
	solution_text_mod = JSON.parse(JSON.stringify(solution_text))
	if (solveIdx >= 0) {
		if (setColor == true) {
			solution_text_mod[solveIdx] = solution_text_mod[solveIdx].bold().fontcolor("red")
		} else {
			solution_text_mod[solveIdx] = solution_text_mod[solveIdx]
		}
	}
	document.getElementById("solution_text").innerHTML = "Solution: <br>"+ solution_text_mod.join(" ");
}

function enableInput() {
	document.getElementById("scramble").disabled=false;
	document.getElementById("solve").disabled=false;
	$(document).on("keypress", buttonPressed);
}

function disableInput() {
	document.getElementById("scramble").disabled=true;
	document.getElementById("solve").disabled=true;
	$(document).off("keypress", buttonPressed);
}

function nextState(moveTimeout=0) {
	if (moves.length > 0) {
		disableInput();
		disableScroll();
		move = moves.shift() // get Move
		
		//convert to python representation
		state_rep = reOrderArray(state,FEToState)
		newState_rep = JSON.parse(JSON.stringify(state_rep))

		//swap stickers
		for (var i = 0; i < rotateIdxs_new[move].length; i++) {
			newState_rep[rotateIdxs_new[move][i]] = state_rep[rotateIdxs_old[move][i]]
		}

		// Change move highlight
		if (moveTimeout != 0){ //check if nextState is used for first_state click, prev_state,etc.
			solveIdx++
			if(solution_text[solveIdx]=="<br>"){
				solveIdx++
			}
			setSolnText(setColor=true)
		}

		//convert back to HTML representation
		newState = reOrderArray(newState_rep,stateToFE)

		//set new state
		setStickerColors(newState)
		finalState = newState

		//Call again if there are more moves
		if (moves.length > 0) {
			setTimeout(function(){nextState(moveTimeout)}, moveTimeout);
		} else {
			enableInput();
			if (solveMoves.length > 0) {
				enableScroll();
				setSolnText();
			}
		}
	} else {
		enableInput();
		if (solveMoves.length > 0) {
			enableScroll();
			setSolnText();
		}
	}
}

function scrambleCube() {
	disableInput();
	clearSoln();

	numMoves = randInt(100,200);
	for (var i = 0; i < numMoves; i++) {
		moves.push(legalMoves[randInt(0,legalMoves.length)]);
	}

	nextState(0);
}

function solveCube() {
	disableInput();
	clearSoln();
	document.getElementById("solution_text").innerHTML = "SOLVING..."
	$.ajax({
		url: '../basic/solve/',
		data: {"state": JSON.stringify(state)},
		type: 'POST',
		dataType: 'json',
		success: function(response) {
			solveStartState = JSON.parse(JSON.stringify(state))
			solveMoves = response["moves"];
			solveMoves_rev = response["moves_rev"];
			solution_text = response["solve_text"];
			solution_text.push("SOLVED!")
			setSolnText(true);

			moves = JSON.parse(JSON.stringify(solveMoves))

			setTimeout(function(){nextState(500)}, 500);
		},
		error: function(error) {
				console.log(error);
				document.getElementById("solution_text").innerHTML = "..."
				// setTimeout(function(){solveCube()}, 500);
		},
	});
}

function nextStateTest(moveTimeout=0) {
	if (moves.length > 0) {
		disableInput();
		disableScroll();
		move = moves.shift() // get Move
		
		//convert to python representation
		state_rep = reOrderArray(state,FEToState)
		newState_rep = JSON.parse(JSON.stringify(state_rep))

		//swap stickers
		for (var i = 0; i < rotateIdxs_new[move].length; i++) {
			newState_rep[rotateIdxs_new[move][i]] = state_rep[rotateIdxs_old[move][i]]
		}

		// Change move highlight
		if (moveTimeout != 0){ //check if nextState is used for first_state click, prev_state,etc.
			solveIdx++
			if(solution_text[solveIdx]=="<br>"){
				solveIdx++
			}
			setSolnText(setColor=true)
		}

		//convert back to HTML representation
		newState = reOrderArray(newState_rep,stateToFE)

		//set new state
		setStickerColors(newState)
		finalState = newState

		//Call again if there are more moves
		if (moves.length > 0) {
			nextStateTest(moveTimeout);
		} else {
			enableInput();
			if (solveMoves.length > 0) {
				enableScroll();
				setSolnText();
			}
		}
	} else {
		enableInput();
		if (solveMoves.length > 0) {
			enableScroll();
			setSolnText();
		}
	}
}

async function scrambleCubeTest() {
	return new Promise(function(resolve, reject) {
		disableInput();
		clearSoln();

		numMoves = randInt(100,200);
		for (var i = 0; i < numMoves; i++) {
			moves.push(legalMoves[randInt(0,legalMoves.length)]);
		}

		// nextStateTest(0);
		resolve()
	});
}

async function solveCubeTest() {
	disableInput();
	clearSoln();
	document.getElementById("solution_text").innerHTML = "SOLVING..."
	return new Promise(function(resolve, reject) {
		$.ajax({
			url: '../basic/solve/',
			data: {"state": JSON.stringify(state)},
			type: 'POST',
			dataType: 'json',
			success: function(response) {
				solveStartState = JSON.parse(JSON.stringify(state))
				solveMoves = response["moves"];
				solveMoves_rev = response["moves_rev"];
				solution_text = response["solve_text"];
				solution_text.push("SOLVED!")
				setSolnText(true);
	
				moves = JSON.parse(JSON.stringify(solveMoves))
	
				resolve()
			},
			error: function(error) {
				console.log(error);
				document.getElementById("solution_text").innerHTML = "..."
				solveCubeTest();
				reject()
			},
		});
	}

	).catch((e) => {})
}

function openRecog()
{
	window.open("recog");
}

function judgeFinalState(finalState) {
	let flag = true;
	for (let face = 0; face<6 && flag; face++) {
		let res = Math.floor(finalState[face*9 + 0] / 9);
		for (let block=1; block<9 && flag; block++) {
			if (res != Math.floor(finalState[face*9 + block] / 9)) {
				flag  = false;
			}
		}
	}
	return flag;
}

$( document ).ready($(function() {
	disableInput();
	clearSoln();
	$.ajax({
		url: '../basic/initState/',
		data: {},
		type: 'POST',
		dataType: 'json',
		success: function(response) {
			setStickerColors(response["state"]);
			rotateIdxs_old = response["rotateIdxs_old"];
			rotateIdxs_new = response["rotateIdxs_new"];
			stateToFE = response["stateToFE"];
			FEToState = response["FEToState"];
			legalMoves = response["legalMoves"]
			enableInput();
		},
		error: function(error) {
			console.log(error);
		},
	});

	$("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)"); //Initial orientation	

	$('#scramble').click(function() {
		scrambleCube()
	});

	$('#solve').click(function() {
		solveCube()
	});

	$('#test').click(async function() {
		console.log("---Scramble Solving Test---")
		let success = 0, failed = 0;
		let avgTime = 0;
		let casenum = 50;
		for(let i = 0;i < casenum; i++){
			disableInput();
			clearSoln();

			numMoves = randInt(100,200);
			for (let i = 0; i < numMoves; i++) {
				moves.push(legalMoves[randInt(0,legalMoves.length)]);
			}
			nextStateTest()

			let begin = Date.now();
			let res = await solveCubeTest();
			let end = Date.now();
			nextStateTest()
			let result = judgeFinalState(finalState);
			if (result) {
				success+=1;
			} else {
				failed+=1;
			}
			avgTime += end-begin;
			console.log("Scramble Case " + i.toString() + ": " + (end - begin).toString() + "ms, "+result.toString())
		}
		avgTime /= casenum;
		console.log(casenum.toString()+" cases finished: success "+success.toString()+", fail "+failed.toString()+", average solving time "+avgTime.toString()+"ms.");
	});
	

	$('#first_state').click(function() {
		if (solveIdx > 0) {
			moves = solveMoves_rev.slice(0, solveIdx).reverse();
			solveIdx = 0;
			nextState();
		}
	});

	$('#prev_state').click(function() {
		if (solveIdx > 0) {
			solveIdx = solveIdx - 1
			moves.push(solveMoves_rev[solveIdx])
			nextState()
		}
	});

	$('#next_state').click(function() {
		if (solveIdx < solveMoves.length) {
			moves.push(solveMoves[solveIdx])
			solveIdx = solveIdx + 1
			nextState()
		}
	});

	$('#last_state').click(function() {
		if (solveIdx < solveMoves.length) {
			moves = solveMoves.slice(solveIdx, solveMoves.length);
			solveIdx = solveMoves.length
			nextState();
		}
	});

	$('#cube_div').on("mousedown", function(ev) {
		lastMouseX = ev.clientX;
		lastMouseY = ev.clientY;
		$('#cube_div').on("mousemove", mouseMoved);
	});
	$('#cube_div').on("mouseup", function() {
		$('#cube_div').off("mousemove", mouseMoved);
	});
	$('#cube_div').on("mouseleave", function() {
		$('#cube_div').off("mousemove", mouseMoved);
	});

	console.log( "ready!" );
}));


function mouseMoved(ev) {
  var deltaX = ev.pageX - lastMouseX;
  var deltaY = ev.pageY - lastMouseY;

  lastMouseX = ev.pageX;
  lastMouseY = ev.pageY;

  rotY += deltaX * 0.2;
  rotX -= deltaY * 0.5;

  $("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)");
}

