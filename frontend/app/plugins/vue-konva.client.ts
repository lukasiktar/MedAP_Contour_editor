import VueKonva from "vue-konva";

export default defineNuxtPlugin((nuxtApp) => {
    // register all <v-*> Konva components globally
    nuxtApp.vueApp.use(VueKonva)
})