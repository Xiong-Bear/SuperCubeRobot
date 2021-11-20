/*

	Main.js

	01. Menu toggle
	02. Top bar height effect
	03. Home content slider
	04. Home background slider
	05. Home background and content slider
	06. Quote slider
	07. Image slider
	08. Services slider
	09. Employee slider
	10. Work slider
	11. Footer promo
	12. Contact form
	13. Scrollto
	14. Magnific popup
	15. Equal height
	16. fitVids

*/


(function(){
	"use strict";

	/* ==================== 01. Menu toggle ==================== */
	$(function(){
		$('#toggle').click(function (e){
			e.stopPropagation();
		});
		$('html').click(function (e){
			if (!$('.toggle').is($(e.target))){
				$('#toggle').prop("checked", false);
			}
		});
	});

/* Minified js in demo only */
$(window).bind("scroll",function(){if($(this).scrollTop()>100){$(".top-bar").removeClass("tb-large").addClass("tb-small")}else{$(".top-bar").removeClass("tb-small").addClass("tb-large")}});$(".home-c-slider").bxSlider({mode:"horizontal",pager:false,controls:true,nextText:'<i class="bs-right fa fa-angle-right"></i>',prevText:'<i class="bs-left fa fa-angle-left"></i>'});$(".home-bg-slider").bxSlider({mode:"fade",auto:true,speed:1e3,pager:false,controls:false,nextText:'<i class="bs-right fa fa-angle-right"></i>',prevText:'<i class="bs-left fa fa-angle-left"></i>'});$(".home-bgc-slider").bxSlider({mode:"fade",pager:true,controls:true,nextText:'<i class="bs-right fa fa-angle-right"></i>',prevText:'<i class="bs-left fa fa-angle-left"></i>'});$(".quote-slider").bxSlider({mode:"horizontal",controls:false,adaptiveHeight:true});$(".img-slider").bxSlider({mode:"fade",pager:true,controls:true,nextText:'<i class="bs-right fa fa-angle-right"></i>',prevText:'<i class="bs-left fa fa-angle-left"></i>'});$(function(){var e=$(".services-slider");e.owlCarousel({pagination:false,navigation:false,items:4,itemsDesktop:[1e3,3],itemsTablet:[600,2],itemsMobile:[321,1]});$(".serv-next").click(function(){e.trigger("owl.next")});$(".serv-prev").click(function(){e.trigger("owl.prev")})});$(function(){var e=$(".employee-slider");e.owlCarousel({pagination:false,navigation:false,items:4,itemsDesktop:[1e3,3],itemsTablet:[600,2],itemsMobile:[321,1]});$(".emp-next").click(function(){e.trigger("owl.next")});$(".emp-prev").click(function(){e.trigger("owl.prev")})});$(function(){var e=$(".work-slider");e.owlCarousel({pagination:false,navigation:false,items:3,itemsDesktop:[1e3,3],itemsTablet:[600,2],itemsMobile:[321,1]});$(".work-next").click(function(){e.trigger("owl.next")});$(".work-prev").click(function(){e.trigger("owl.prev")})});$(".promo-control").click(function(){$(".footer-promo").slideToggle(500);if($(".footer-promo").is(":visible")){$("html, body").animate({scrollTop:$(".footer-promo").offset().top},500)}});$(function(){$("#contactform").submit(function(){var e=$(this).attr("action");$("#message").slideUp(300,function(){$("#message").hide();$("#submit").after('<img src="images/loader.gif" class="loader">').attr("disabled","disabled");$.post(e,{name:$("#name").val(),email:$("#email").val(),phone:$("#phone").val(),subject:$("#subject").val(),comments:$("#comments").val(),verify:$("#verify").val()},function(e){document.getElementById("message").innerHTML=e;$("#message").slideDown(300);$("#contactform img.loader").fadeOut(300,function(){$(this).remove()});$("#submit").removeAttr("disabled");if(e.match("success")!=null)$("#contactform").slideUp(300)})});return false})});$(function(){$(".scrollto").bind("click.scrollto",function(e){e.preventDefault();var t=this.hash,n=$(t);$("html, body").stop().animate({scrollTop:n.offset().top-0},900,"swing",function(){window.location.hash=t})})});$(".popup").magnificPopup({type:"image",fixedContentPos:false,fixedBgPos:false,removalDelay:300,mainClass:"mfp-fade"});$(".popup-youtube, .popup-vimeo, .popup-gmaps").magnificPopup({disableOn:700,type:"iframe",fixedContentPos:false,fixedBgPos:false,removalDelay:300,mainClass:"mfp-fade",preloader:false});$(".popup-gallery").magnificPopup({type:"image",gallery:{enabled:true},fixedContentPos:false,fixedBgPos:false,removalDelay:300,mainClass:"mfp-fade"});$(document).ready(function(){$(".equal").children(".col").equalizeHeight();$(window).resize(function(){$(".equal").children(".col").equalizeHeight()})});$(".responsive-video").fitVids()

})(jQuery);