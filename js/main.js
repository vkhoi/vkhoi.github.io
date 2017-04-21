$(document).ready(function() {
	var imgs = $('img.center-image');
	
	imgs.each(function(index, img) {
		var parent = $(img).parent();
		$(parent).addClass('center');
	});
});